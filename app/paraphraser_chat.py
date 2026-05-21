import json
import re
from typing import Any

from sqlalchemy.orm import Session

from app.crud import get_existing_conversation_for_task
from app.providers import ProviderResult, send_messages_with_model
from app.schemas import (
    ChatMessage,
    CostUsage,
    ParaphraserChatRequest,
    ParaphraserChatResponse,
    ParaphraserResultItem,
    ParaphraserState,
    TokenUsage,
)
from app.settings import get_settings
from app.tasks import (
    PARAPHRASER_EXTRACTOR_REPAIR_PROMPT,
    PARAPHRASER_EXTRACTOR_SYSTEM_PROMPT,
    PARAPHRASER_GENERATOR_CHAT_PROMPT,
)


# Suggestions only. They are not strict backend enums.
SUGGESTED_LANGUAGES = [
    "Auto Detect",
    "Arabic",
    "English",
    "French",
    "Spanish",
    "Turkish",
]

SUGGESTED_TONES = [
    "Professional",
    "Simple",
    "Creative",
    "Marketing",
    "Academic",
    "Human-like",
    "Formal",
    "Neutral",
    "Journalistic",
]

SUGGESTED_REWRITE_MODES = [
    "Paraphrase",
    "Shorter",
    "Longer",
    "Humanize",
    "Improve Clarity",
    "Fix Grammar",
    "Simplify",
    "Make Professional",
]

SUGGESTED_CHANGE_LEVELS = [
    "Low",
    "Medium",
    "High",
]

SUGGESTED_EXTRA_OPTIONS = [
    "Preserve meaning",
    "Do not add facts",
    "Keep keywords",
    "Make more natural",
    "Make more persuasive",
    "SEO-friendly rewrite",
]

MAX_RESULTS_COUNT = 20
REQUIRED_FIELDS = [
    "content",
    "language",
    "tone",
    "rewrite_mode",
    "change_level",
    "results_count",
]


def default_paraphraser_state() -> ParaphraserState:
    return ParaphraserState(
        content=None,
        language="Auto Detect",
        tone="Professional",
        rewrite_mode="Paraphrase",
        change_level="Medium",
        results_count=1,
        extra_options=["Preserve meaning", "Do not add facts"],
    )


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def normalize_arabic_text(value: str) -> str:
    """Light normalization only; does not remove meaning or punctuation."""
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_extracted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # IMPORTANT: The AI extractor must not return the long user content.
    # Long paraphrasing content is extracted with Python in extract_paraphraser_content()
    # to avoid invalid/truncated JSON when the user sends large articles.
    clean = {
        "language": normalize_text(payload.get("language")),
        "tone": normalize_text(payload.get("tone")),
        "rewrite_mode": normalize_text(payload.get("rewrite_mode")),
        "change_level": normalize_text(payload.get("change_level")),
        "results_count": payload.get("results_count"),
        "extra_options": payload.get("extra_options") or [],
    }

    try:
        if clean["results_count"] is not None and clean["results_count"] != "":
            count = int(clean["results_count"])
            clean["results_count"] = count if 1 <= count <= MAX_RESULTS_COUNT else None
        else:
            clean["results_count"] = None
    except (TypeError, ValueError):
        clean["results_count"] = None

    if not isinstance(clean["extra_options"], list):
        clean["extra_options"] = [clean["extra_options"]]

    options: list[str] = []
    for option in clean["extra_options"]:
        option = str(option).strip()
        if option and option not in options:
            options.append(option)
    clean["extra_options"] = options

    return clean


def extract_paraphraser_content(user_message: str) -> tuple[str | None, str]:
    """
    Extract the long text to rewrite using code, not AI.

    The extractor model should only receive the user's instruction/settings, not
    the entire article. This prevents truncated invalid JSON for long Arabic or
    English inputs.
    """
    text = (user_message or "").strip()
    if not text:
        return None, text

    # Common case: instruction followed by quoted long text.
    quote_patterns = [
        r"[:：]\s*['\"“”](.{80,})['\"“”]\s*$",
        r"['\"“”](.{120,})['\"“”]\s*$",
    ]
    for pattern in quote_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            content = normalize_arabic_text(match.group(1).strip().strip("'\"“”"))
            instruction_text = text[: match.start(1)].strip() + " [CONTENT_REMOVED]"
            return content, instruction_text.strip()

    # Common case: instruction followed by colon and long text without quotes.
    instruction_markers = [
        "أعد صياغة هذا النص",
        "اعد صياغة هذا النص",
        "إعادة صياغة",
        "اعادة صياغة",
        "أعد صياغة",
        "اعد صياغة",
        "paraphrase this text",
        "rewrite this text",
        "paraphrase",
        "rewrite",
    ]

    lower_text = text.lower()
    if any(marker.lower() in lower_text for marker in instruction_markers):
        parts = re.split(r"[:：]", text, maxsplit=1)
        if len(parts) == 2 and len(parts[1].strip()) >= 80:
            content = normalize_arabic_text(parts[1].strip().strip("'\"“”"))
            instruction_text = parts[0].strip() + ": [CONTENT_REMOVED]"
            return content, instruction_text

    # If the user pasted a long text only, treat the whole message as content.
    # Follow-up edit messages are usually short, so they will not hit this branch.
    if len(text) >= 350:
        return normalize_arabic_text(text), "[USER_SENT_CONTENT_ONLY]"

    return None, text


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Extractor JSON must be an object")
        return parsed
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Extractor did not return valid JSON: {text}")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Extractor JSON must be an object")
        return parsed


def merge_extra_options(old_options: list[str], new_options: list[str] | None) -> list[str]:
    merged: list[str] = []
    for item in old_options + (new_options or []):
        item = str(item).strip()
        if item and item not in merged:
            merged.append(item)
    return merged


def merge_paraphraser_state(old_state: ParaphraserState, extracted: dict[str, Any]) -> ParaphraserState:
    """
    The extractor returns the FULL updated state. This function still protects old values
    if the extractor accidentally returns null for a saved value.
    """
    data = old_state.model_dump()

    # The extractor updates only settings/options. Content is handled separately
    # by extract_paraphraser_content() to avoid returning long text in JSON.
    for key in ["language", "tone", "rewrite_mode", "change_level", "results_count"]:
        value = extracted.get(key)
        if value is not None and value != "":
            data[key] = value

    data["extra_options"] = merge_extra_options(
        old_state.extra_options,
        extracted.get("extra_options") or [],
    )

    return ParaphraserState(**data)


def is_ready_for_generation(state: ParaphraserState) -> bool:
    return bool((state.content or "").strip())


def get_missing_fields(state: ParaphraserState) -> list[str]:
    missing: list[str] = []
    if not (state.content or "").strip():
        missing.append("content")
    return missing


def get_question_message(state: ParaphraserState) -> str:
    content = state.content or ""
    looks_arabic = bool(re.search(r"[\u0600-\u06FF]", content))
    is_arabic = state.language == "Arabic" or state.language == "Auto Detect" or looks_arabic
    if is_arabic:
        return "ما النص الذي تريد إعادة صياغته؟"
    return "What text would you like me to paraphrase?"


def build_extractor_user_prompt(state: ParaphraserState, user_message: str) -> str:
    state_without_content = state.model_dump()
    if state_without_content.get("content"):
        state_without_content["content"] = "[CONTENT_ALREADY_SAVED]"

    return f"""
Current saved state, with long content hidden:
{json.dumps(state_without_content, ensure_ascii=False)}

Latest user instruction/message, with long content removed when present:
{user_message}

Return updated settings as JSON only. Do NOT include the article/text content.

Allowed JSON schema:
{{
  "language": null,
  "tone": null,
  "rewrite_mode": null,
  "change_level": null,
  "results_count": null,
  "extra_options": []
}}

Rules:
- Return only the small settings JSON above.
- NEVER return the full text/article inside JSON.
- Keep old state values unless the user clearly changes them.
- If the user says "خليه", "اجعله", "make it", "make this", "shorten it", "humanize it", update only the requested options.
- language examples = {SUGGESTED_LANGUAGES}. Accept any language requested by the user.
- tone examples = {SUGGESTED_TONES}. Accept any tone requested by the user.
- rewrite_mode examples = {SUGGESTED_REWRITE_MODES}. Accept any mode requested by the user.
- change_level examples = {SUGGESTED_CHANGE_LEVELS}. Accept any descriptive level requested by the user.
- results_count examples = [1, 2, 3, 5]. Accept any positive integer up to {MAX_RESULTS_COUNT}.
- extra_options examples = {SUGGESTED_EXTRA_OPTIONS}.

Arabic mapping rules:
- "احترافي" => tone = "Professional", rewrite_mode = "Make Professional".
- "بسيط" or "سهل" => tone = "Simple", rewrite_mode = "Simplify".
- "بشري" or "طبيعي" => tone = "Human-like", rewrite_mode = "Humanize".
- "أقصر" or "اختصر" => rewrite_mode = "Shorter".
- "أطول" or "وسع" => rewrite_mode = "Longer".
- "تسويقي" => tone = "Marketing".
- "أكاديمي" => tone = "Academic".
- "فصحى" => language = "Arabic" and tone = "Formal".
- Arabic instruction => language = "Arabic" unless another language is requested.
- If the user asks for a specific number of versions/results, use that exact number.
""".strip()


async def extract_paraphraser_updates_with_retry(
    state: ParaphraserState,
    user_message: str,
):
    extractor_messages = [
        ChatMessage(role="system", content=PARAPHRASER_EXTRACTOR_SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_extractor_user_prompt(state, user_message)),
    ]

    extractor_result = await send_messages_with_model(
        model_key="paraphraser_extractor",
        messages=extractor_messages,
        temperature_override=0.0,
        max_tokens_override=500,
        enable_web_search=False,
        response_format={"type": "json_object"},
    )

    try:
        extracted = normalize_extracted_payload(extract_json_object(extractor_result.content))
        return extracted, extractor_result, None
    except Exception as first_error:
        repair_messages = [
            ChatMessage(role="system", content=PARAPHRASER_EXTRACTOR_REPAIR_PROMPT),
            ChatMessage(
                role="user",
                content=f"""
Current saved state, with long content hidden:
{json.dumps({**state.model_dump(), "content": "[CONTENT_ALREADY_SAVED]" if state.content else None}, ensure_ascii=False)}

Latest user instruction/message, with long content removed when present:
{user_message}

Invalid extractor output:
{extractor_result.content}

Return corrected settings JSON only. Do NOT include content.
""".strip(),
            ),
        ]

        repair_result = await send_messages_with_model(
            model_key="paraphraser_extractor",
            messages=repair_messages,
            temperature_override=0.0,
            max_tokens_override=500,
            enable_web_search=False,
            response_format={"type": "json_object"},
        )

        try:
            extracted = normalize_extracted_payload(extract_json_object(repair_result.content))
            return extracted, repair_result, {
                "first_error": str(first_error),
                "first_raw": extractor_result.content,
                "repaired": True,
            }
        except Exception as second_error:
            raise ValueError(
                "Extractor failed to return valid JSON after retry. "
                f"First error: {first_error}. "
                f"Second error: {second_error}. "
                f"First raw output: {extractor_result.content}. "
                f"Repair raw output: {repair_result.content}."
            )


def calculate_dynamic_max_tokens(content: str, results_count: int, rewrite_mode: str | None) -> int:
    word_count = len((content or "").split())
    multiplier = 2.2
    if rewrite_mode and rewrite_mode.lower() in {"shorter", "اختصار", "مختصر"}:
        multiplier = 1.2
    elif rewrite_mode and rewrite_mode.lower() in {"longer", "expand", "أطول"}:
        multiplier = 3.0

    estimated = int(max(500, word_count * multiplier * max(1, results_count)))
    return min(3500, estimated)


def build_generator_user_prompt(state: ParaphraserState) -> str:
    return f"""
Rewrite the text below and return exactly {state.results_count} version(s).

Text:
{state.content}

Language:
{state.language}

Tone:
{state.tone}

Rewrite Mode:
{state.rewrite_mode}

Change Level:
{state.change_level}

Extra Options:
{", ".join(state.extra_options) if state.extra_options else "None"}

Output requirements:
- Preserve the original meaning and facts.
- Do not add unsupported information.
- Do not remove important details.
- Make the wording natural, clear, and ready to use.
- If multiple versions are requested, make them meaningfully different.
- Return only the rewritten version(s).
- Number the versions when more than one result is requested.
""".strip()


def clean_result_text(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*(?:version|result|option)\s*\d+\s*[:.)-]\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\s*(?:النسخة|النتيجة|الخيار)\s*\d+\s*[:.)-]\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\s*\d+\s*[\).\-:]\s*", "", line)
    line = re.sub(r"^\s*[-•]\s*", "", line)
    return line.strip(" \t\r\n\"'“”")


def parse_paraphraser_results(text: str, requested_count: int) -> list[ParaphraserResultItem]:
    text = (text or "").strip()
    if requested_count <= 1:
        cleaned = clean_result_text(text)
        return [ParaphraserResultItem(id=1, text=cleaned)] if cleaned else []

    # Split on numbered version markers while keeping paragraphs inside each version.
    pattern = r"(?:^|\n)\s*(?:\d+\s*[\).:-]|(?:Version|Result|Option)\s*\d+\s*[:.)-]|(?:النسخة|النتيجة|الخيار)\s*\d+\s*[:.)-])\s*"
    parts = [part.strip() for part in re.split(pattern, text, flags=re.IGNORECASE) if part.strip()]

    if len(parts) < requested_count:
        lines = [clean_result_text(line) for line in text.splitlines() if clean_result_text(line)]
        parts = lines if len(lines) >= requested_count else parts

    if not parts:
        parts = [text]

    results: list[ParaphraserResultItem] = []
    for part in parts:
        cleaned = clean_result_text(part)
        if not cleaned:
            continue
        results.append(ParaphraserResultItem(id=len(results) + 1, text=cleaned))
        if len(results) >= requested_count:
            break

    return results


def calculate_paraphraser_cost(input_tokens: int | None, output_tokens: int | None) -> CostUsage:
    settings = get_settings()

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0

    input_cost = (input_tokens / 1_000_000) * settings.PARAPHRASER_INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * settings.PARAPHRASER_OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost

    return CostUsage(
        input_cost=round(input_cost, 8),
        output_cost=round(output_cost, 8),
        web_search_cost=0,
        total_cost=round(total_cost, 8),
        currency="USD",
    )


def combine_usage(*items: ProviderResult) -> TokenUsage:
    input_tokens = sum((item.input_tokens or 0) for item in items if item is not None)
    output_tokens = sum((item.output_tokens or 0) for item in items if item is not None)
    total_tokens = sum((item.total_tokens or 0) for item in items if item is not None)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
    )


async def run_paraphraser_chat(
    db: Session,
    req: ParaphraserChatRequest,
    request_id: str,
) -> ParaphraserChatResponse:
    settings = get_settings()

    get_existing_conversation_for_task(
        db=db,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
    )

    state = req.state or default_paraphraser_state()

    if len(req.user_message) > settings.MAX_USER_MESSAGE_LENGTH:
        raise ValueError(f"user_message exceeds max length of {settings.MAX_USER_MESSAGE_LENGTH}")

    detected_content, instruction_message = extract_paraphraser_content(req.user_message)

    extracted, extractor_result, extractor_repair_debug = await extract_paraphraser_updates_with_retry(
        state=state,
        user_message=instruction_message,
    )

    new_state = merge_paraphraser_state(state, extracted)

    if detected_content:
        new_state.content = detected_content

    if new_state.content and len(new_state.content) > settings.PARAPHRASER_MAX_CONTENT_CHARS:
        new_state.content = new_state.content[: settings.PARAPHRASER_MAX_CONTENT_CHARS].strip()

    if not is_ready_for_generation(new_state):
        usage = combine_usage(extractor_result)
        cost = calculate_paraphraser_cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        debug = None
        if req.debug and settings.ENABLE_DEBUG_RESPONSE:
            debug = {
                "phase": "question",
                "extracted": extracted,
                "extractor_raw": extractor_result.content,
                "instruction_message": instruction_message,
                "detected_content": bool(detected_content),
                "repair": extractor_repair_debug,
                "state": new_state.model_dump(),
                "missing": get_missing_fields(new_state),
            }

        return ParaphraserChatResponse(
            type="question",
            user_id=req.user_id,
            sub_tool_id=req.sub_tool_id,
            conversation_uuid=req.conversation_uuid,
            message=get_question_message(new_state),
            state=new_state,
            results=[],
            count=0,
            request_id=request_id,
            debug=debug,
            usage=usage,
            cost=cost,
        )

    results_count = new_state.results_count or 1
    max_tokens = calculate_dynamic_max_tokens(
        content=new_state.content or "",
        results_count=results_count,
        rewrite_mode=new_state.rewrite_mode,
    )

    generator_messages = [
        ChatMessage(role="system", content=PARAPHRASER_GENERATOR_CHAT_PROMPT),
        ChatMessage(role="user", content=build_generator_user_prompt(new_state)),
    ]

    generator_result = await send_messages_with_model(
        model_key="paraphraser_fast",
        messages=generator_messages,
        temperature_override=0.45,
        max_tokens_override=max_tokens,
        enable_web_search=False,
    )

    results = parse_paraphraser_results(generator_result.content, results_count)

    usage = combine_usage(extractor_result, generator_result)
    cost = calculate_paraphraser_cost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )

    is_arabic = new_state.language == "Arabic" or bool(re.search(r"[\u0600-\u06FF]", new_state.content or ""))
    message = "تمت إعادة صياغة النص بنجاح." if is_arabic else "Text paraphrased successfully."

    debug = None
    if req.debug and settings.ENABLE_DEBUG_RESPONSE:
        debug = {
            "phase": "result",
            "extracted": extracted,
            "extractor_raw": extractor_result.content,
            "instruction_message": instruction_message,
            "detected_content": bool(detected_content),
            "repair": extractor_repair_debug,
            "generator_raw": generator_result.content,
            "state": new_state.model_dump(),
            "results_count": len(results),
            "max_tokens": max_tokens,
        }

    return ParaphraserChatResponse(
        type="result",
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
        message=message,
        state=new_state,
        results=results,
        count=len(results),
        request_id=request_id,
        debug=debug,
        usage=usage,
        cost=cost,
    )
