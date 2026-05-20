import json
import re
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy.orm import Session

from app.crud import get_existing_conversation_for_task
from app.providers import ProviderResult, send_messages_with_model
from app.schemas import (
    ChatMessage,
    CostUsage,
    HeadlineChatRequest,
    HeadlineChatResponse,
    HeadlineItem,
    HeadlineState,
    TokenUsage,
)
from app.settings import get_settings
from app.tasks import (
    HEADLINE_EXTRACTOR_REPAIR_PROMPT,
    HEADLINE_EXTRACTOR_SYSTEM_PROMPT,
    HEADLINE_GENERATOR_CHAT_PROMPT,
)


VALID_CONTENT_TYPES = {
    "Article",
    "News",
    "YouTube Video",
    "Social Media Post",
    "Ad",
    "Email Subject",
    "Landing Page",
    "Product",
    "Report",
    "Creative Text",
    "General",
}

VALID_GOALS = {
    "Attract Attention",
    "Explain Clearly",
    "Increase Clicks",
    "Sound Professional",
    "Sound Creative",
    "Improve SEO",
    "Create Curiosity",
    "Sell / Convert",
    "Summarize Content",
}

VALID_LANGUAGES = {
    "Auto Detect",
    "Arabic",
    "English",
    "French",
    "Chinese",
    "Russian",
}

VALID_TONES = {
    "Professional",
    "Powerful",
    "Simple",
    "Creative",
    "Emotional",
    "Luxury",
    "Bold",
    "Informative",
    "Journalistic",
    "Academic",
    "Marketing",
    "Neutral",
}

VALID_HEADLINE_LENGTHS = {
    "Short",
    "Medium",
    "Long",
    "Auto",
}

VALID_COUNTS = {5, 10, 15, 20}

VALID_EXTRA_OPTIONS = {
    "Include SEO-friendly headlines",
    "Include curiosity-based headlines",
    "Include professional headlines",
    "Avoid clickbait",
    "Avoid exaggeration",
    "Generate headline + subheadline",
}

REQUIRED_FIELDS = [
    "content",
    "content_type",
    "goal",
    "language",
    "tone",
    "number_of_headlines",
    "headline_length",
]


def default_headline_state() -> HeadlineState:
    return HeadlineState(
        content=None,
        content_type=None,
        goal=None,
        language=None,
        tone=None,
        number_of_headlines=None,
        headline_length=None,
        extra_options=[],
    )


def normalize_extracted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    clean = {
        "content": payload.get("content"),
        "content_type": payload.get("content_type"),
        "goal": payload.get("goal"),
        "language": payload.get("language"),
        "tone": payload.get("tone"),
        "number_of_headlines": payload.get("number_of_headlines"),
        "headline_length": payload.get("headline_length"),
        "extra_options": payload.get("extra_options") or [],
    }

    if clean["content"] is not None:
        clean["content"] = str(clean["content"]).strip() or None

    if clean["content_type"] not in VALID_CONTENT_TYPES:
        clean["content_type"] = None

    if clean["goal"] not in VALID_GOALS:
        clean["goal"] = None

    if clean["language"] not in VALID_LANGUAGES:
        clean["language"] = None

    if clean["tone"] not in VALID_TONES:
        clean["tone"] = None

    if clean["headline_length"] not in VALID_HEADLINE_LENGTHS:
        clean["headline_length"] = None

    try:
        if clean["number_of_headlines"] is not None:
            clean["number_of_headlines"] = int(clean["number_of_headlines"])
    except (TypeError, ValueError):
        clean["number_of_headlines"] = None

    if clean["number_of_headlines"] not in VALID_COUNTS:
        clean["number_of_headlines"] = None

    if not isinstance(clean["extra_options"], list):
        clean["extra_options"] = []

    clean["extra_options"] = [
        option
        for option in clean["extra_options"]
        if option in VALID_EXTRA_OPTIONS
    ]

    return clean


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
        if item in VALID_EXTRA_OPTIONS and item not in merged:
            merged.append(item)
    return merged


def merge_headline_state(old_state: HeadlineState, extracted: dict[str, Any]) -> HeadlineState:
    """
    The extractor returns the FULL updated state. This function still protects old values
    if the extractor accidentally returns null for a value that was already saved.
    """
    data = old_state.model_dump()

    for key in REQUIRED_FIELDS:
        value = extracted.get(key)
        if value is not None and value != "":
            data[key] = value

    # If content is intentionally changed to a new non-empty value, it is handled above.
    # Keep old content when extractor returns null.
    data["extra_options"] = merge_extra_options(
        old_state.extra_options,
        extracted.get("extra_options") or [],
    )

    return HeadlineState(**data)


def is_ready_for_generation(state: HeadlineState) -> bool:
    return all(
        [
            bool((state.content or "").strip()),
            state.content_type is not None,
            state.goal is not None,
            state.language is not None,
            state.tone is not None,
            state.number_of_headlines is not None,
            state.headline_length is not None,
        ]
    )


def get_missing_fields(state: HeadlineState) -> list[str]:
    missing: list[str] = []

    if not (state.content or "").strip():
        missing.append("content")
    if state.content_type is None:
        missing.append("content_type")
    if state.goal is None:
        missing.append("goal")
    if state.language is None:
        missing.append("language")
    if state.tone is None:
        missing.append("tone")
    if state.number_of_headlines is None:
        missing.append("number_of_headlines")
    if state.headline_length is None:
        missing.append("headline_length")

    return missing


def get_question_message(state: HeadlineState) -> str:
    missing = get_missing_fields(state)
    if not missing:
        return "All required data is complete."

    # If Arabic is detected or still unknown but content looks Arabic, ask in Arabic.
    content = state.content or ""
    looks_arabic = bool(re.search(r"[\u0600-\u06FF]", content))
    is_arabic = state.language == "Arabic" or (state.language is None and looks_arabic)

    labels_ar = {
        "content": "ما هو النص أو الموضوع الذي تريد توليد عناوين له؟",
        "content_type": "ما نوع المحتوى؟ اختر واحدًا: Article, News, YouTube Video, Social Media Post, Ad, Email Subject, Landing Page, Product, Report, Creative Text, General",
        "goal": "ما الهدف من العناوين؟ اختر واحدًا: Attract Attention, Explain Clearly, Increase Clicks, Sound Professional, Sound Creative, Improve SEO, Create Curiosity, Sell / Convert, Summarize Content",
        "language": "ما اللغة المطلوبة؟ اختر واحدة: Auto Detect, Arabic, English, French, Chinese, Russian",
        "tone": "ما النبرة المطلوبة؟ اختر واحدة: Professional, Powerful, Simple, Creative, Emotional, Luxury, Bold, Informative, Journalistic, Academic, Marketing, Neutral",
        "number_of_headlines": "كم عدد العناوين؟ اختر واحدًا: 5, 10, 15, 20",
        "headline_length": "ما طول العنوان؟ اختر واحدًا: Short, Medium, Long, Auto",
    }

    labels_en = {
        "content": "What is the content, topic, product, or idea you want headlines for?",
        "content_type": "What is the content type? Choose one: Article, News, YouTube Video, Social Media Post, Ad, Email Subject, Landing Page, Product, Report, Creative Text, General",
        "goal": "What is the goal? Choose one: Attract Attention, Explain Clearly, Increase Clicks, Sound Professional, Sound Creative, Improve SEO, Create Curiosity, Sell / Convert, Summarize Content",
        "language": "What language do you want? Choose one: Auto Detect, Arabic, English, French, Chinese, Russian",
        "tone": "What tone do you want? Choose one: Professional, Powerful, Simple, Creative, Emotional, Luxury, Bold, Informative, Journalistic, Academic, Marketing, Neutral",
        "number_of_headlines": "How many headlines do you want? Choose one: 5, 10, 15, 20",
        "headline_length": "What headline length do you want? Choose one: Short, Medium, Long, Auto",
    }

    labels = labels_ar if is_arabic else labels_en
    return labels[missing[0]]


def build_extractor_user_prompt(state: HeadlineState, user_message: str) -> str:
    return f"""
Current saved state:
{json.dumps(state.model_dump(), ensure_ascii=False)}

Latest user message:
{user_message}

Task:
Update the current saved state using only the latest user message.
Return the FULL updated JSON state, not only changed fields.

Very important rules:
- Return valid JSON only.
- Do not explain.
- Do not include markdown.
- Do not include analysis.
- First character must be {{ and last character must be }}.
- Keep old non-null values from Current saved state unless the latest user message clearly changes them.
- Use null only for fields that are still unknown.
- extra_options must always be an array.
- Do not generate headlines.

Required JSON shape:
{{
  "content": null,
  "content_type": null,
  "goal": null,
  "language": null,
  "tone": null,
  "number_of_headlines": null,
  "headline_length": null,
  "extra_options": []
}}

Allowed values:
content_type = ["Article", "News", "YouTube Video", "Social Media Post", "Ad", "Email Subject", "Landing Page", "Product", "Report", "Creative Text", "General"]
goal = ["Attract Attention", "Explain Clearly", "Increase Clicks", "Sound Professional", "Sound Creative", "Improve SEO", "Create Curiosity", "Sell / Convert", "Summarize Content"]
language = ["Auto Detect", "Arabic", "English", "French", "Chinese", "Russian"]
tone = ["Professional", "Powerful", "Simple", "Creative", "Emotional", "Luxury", "Bold", "Informative", "Journalistic", "Academic", "Marketing", "Neutral"]
number_of_headlines = [5, 10, 15, 20]
headline_length = ["Short", "Medium", "Long", "Auto"]
extra_options = ["Include SEO-friendly headlines", "Include curiosity-based headlines", "Include professional headlines", "Avoid clickbait", "Avoid exaggeration", "Generate headline + subheadline"]

Arabic mapping rules:
- "مقال", "المقال", "لهذا المقال", "لهزا المقال" => content_type = "Article".
- "خبر" => content_type = "News".
- "عنوان" or "عناوين" means the user wants headline generation; do not put the word itself as content.
- "قوي", "ملفت", "جذاب" => tone = "Powerful" and goal = "Attract Attention".
- "احترافي" => tone = "Professional" and add "Include professional headlines".
- "سيو" or "SEO" => goal = "Improve SEO" and add "Include SEO-friendly headlines".
- Arabic text => language = "Arabic" unless another language is requested.
- If the user asks for one headline but allowed numbers are only [5, 10, 15, 20], set number_of_headlines = 5.
- If the user gives a quoted phrase or topic after words like "عن", "حول", "لهذا المقال", "لهزا المقال", use that phrase/topic as content.
""".strip()


async def extract_headline_updates_with_retry(
    state: HeadlineState,
    user_message: str,
):
    extractor_messages = [
        ChatMessage(role="system", content=HEADLINE_EXTRACTOR_SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_extractor_user_prompt(state, user_message)),
    ]

    extractor_result = await send_messages_with_model(
        model_key="headline_extractor",
        messages=extractor_messages,
        temperature_override=0.0,
        max_tokens_override=700,
        enable_web_search=False,
        response_format={"type": "json_object"},
    )

    try:
        extracted = normalize_extracted_payload(extract_json_object(extractor_result.content))
        return extracted, extractor_result, None
    except Exception as first_error:
        repair_messages = [
            ChatMessage(role="system", content=HEADLINE_EXTRACTOR_REPAIR_PROMPT),
            ChatMessage(
                role="user",
                content=f"""
Current saved state:
{json.dumps(state.model_dump(), ensure_ascii=False)}

Latest user message:
{user_message}

Invalid extractor output:
{extractor_result.content}

Return the corrected FULL updated JSON state only.
""".strip(),
            ),
        ]

        repair_result = await send_messages_with_model(
            model_key="headline_extractor",
            messages=repair_messages,
            temperature_override=0.0,
            max_tokens_override=700,
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


def build_generator_user_prompt(state: HeadlineState) -> str:
    return f"""
Generate exactly {state.number_of_headlines} headline options.

Input:
{state.content}

Content Type:
{state.content_type}

Goal:
{state.goal}

Tone:
{state.tone}

Language:
{state.language}

Headline Length:
{state.headline_length}

Extra Options:
{", ".join(state.extra_options) if state.extra_options else "None"}

Requirements:
- Make every headline distinct.
- Avoid generic wording.
- Avoid misleading or exaggerated claims.
- Match the selected content type and goal.
- Return only the final headlines.
""".strip()


def clean_headline_text(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*\d+[\).\-\:]\s*", "", line)
    line = re.sub(r"^\s*[-•]\s*", "", line)
    line = line.strip(" \t\r\n\"'“”")

    prefixes = ["Headline:", "العنوان:", "Title:"]
    for prefix in prefixes:
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix):].strip()

    return line.strip()


def is_too_similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.80


def parse_headlines(text: str, requested_count: int) -> list[HeadlineItem]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    bad_prefixes = [
        "here are",
        "بالطبع",
        "إليك",
        "هذه",
        "headlines",
    ]

    cleaned: list[str] = []

    for line in lines:
        candidate = clean_headline_text(line)
        if not candidate:
            continue

        lower = candidate.lower()
        if any(lower.startswith(prefix) for prefix in bad_prefixes):
            continue

        if len(candidate) < 4:
            continue

        duplicate = any(candidate == old or is_too_similar(candidate, old) for old in cleaned)
        if duplicate:
            continue

        cleaned.append(candidate)
        if len(cleaned) >= requested_count:
            break

    return [HeadlineItem(id=index + 1, text=headline) for index, headline in enumerate(cleaned)]


def calculate_headline_cost(input_tokens: int | None, output_tokens: int | None) -> CostUsage:
    settings = get_settings()

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0

    input_cost = (input_tokens / 1_000_000) * settings.HEADLINE_INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * settings.HEADLINE_OUTPUT_COST_PER_1M
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


async def run_headline_chat(
    db: Session,
    req: HeadlineChatRequest,
    request_id: str,
) -> HeadlineChatResponse:
    settings = get_settings()

    get_existing_conversation_for_task(
        db=db,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
    )

    state = req.state or default_headline_state()

    if len(req.user_message) > settings.MAX_USER_MESSAGE_LENGTH:
        raise ValueError(f"user_message exceeds max length of {settings.MAX_USER_MESSAGE_LENGTH}")

    extracted, extractor_result, extractor_repair_debug = await extract_headline_updates_with_retry(
        state=state,
        user_message=req.user_message,
    )

    new_state = merge_headline_state(state, extracted)

    if new_state.content and len(new_state.content) > settings.HEADLINE_MAX_CONTENT_CHARS:
        new_state.content = new_state.content[: settings.HEADLINE_MAX_CONTENT_CHARS].strip()

    if not is_ready_for_generation(new_state):
        usage = combine_usage(extractor_result)
        cost = calculate_headline_cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

        debug = None
        if req.debug and settings.ENABLE_DEBUG_RESPONSE:
            debug = {
                "phase": "question",
                "extracted": extracted,
                "extractor_raw": extractor_result.content,
                "repair": extractor_repair_debug,
                "state": new_state.model_dump(),
                "missing": get_missing_fields(new_state),
            }

        return HeadlineChatResponse(
            type="question",
            user_id=req.user_id,
            sub_tool_id=req.sub_tool_id,
            conversation_uuid=req.conversation_uuid,
            message=get_question_message(new_state),
            state=new_state,
            headlines=[],
            count=0,
            request_id=request_id,
            debug=debug,
            usage=usage,
            cost=cost,
        )

    generator_messages = [
        ChatMessage(role="system", content=HEADLINE_GENERATOR_CHAT_PROMPT),
        ChatMessage(role="user", content=build_generator_user_prompt(new_state)),
    ]

    generator_result = await send_messages_with_model(
        model_key="headline_fast",
        messages=generator_messages,
        temperature_override=0.75,
        max_tokens_override=1200,
        enable_web_search=False,
    )

    requested_count = new_state.number_of_headlines or 5
    headlines = parse_headlines(generator_result.content, requested_count)

    usage = combine_usage(extractor_result, generator_result)
    cost = calculate_headline_cost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )

    message = "تم توليد العناوين بنجاح." if new_state.language == "Arabic" else "Headlines generated successfully."

    debug = None
    if req.debug and settings.ENABLE_DEBUG_RESPONSE:
        debug = {
            "phase": "result",
            "extracted": extracted,
            "extractor_raw": extractor_result.content,
            "repair": extractor_repair_debug,
            "generator_raw": generator_result.content,
            "state": new_state.model_dump(),
            "headlines_count": len(headlines),
        }

    return HeadlineChatResponse(
        type="result",
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
        message=message,
        state=new_state,
        headlines=headlines,
        count=len(headlines),
        request_id=request_id,
        debug=debug,
        usage=usage,
        cost=cost,
    )
