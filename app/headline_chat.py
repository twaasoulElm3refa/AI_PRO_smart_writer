import json
import re
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy.orm import Session

from app.crud import get_existing_conversation_for_task
from app.content_common import combine_usage, merge_extra_options, normalize_text
from app.errors import ProviderOutputError
from app.json_utils import extract_json_object, object_response_format
from app.providers import send_messages_with_model
from app.schemas import (
    ChatMessage,
    CostUsage,
    HeadlineChatRequest,
    HeadlineChatResponse,
    HeadlineItem,
    HeadlineState,
)
from app.settings import get_settings
from app.tasks import (
    HEADLINE_EXTRACTOR_REPAIR_PROMPT,
    HEADLINE_EXTRACTOR_SYSTEM_PROMPT,
    HEADLINE_GENERATOR_CHAT_PROMPT,
)


# These lists are suggestions for the UI and for the model prompt only.
# They are NOT strict validation lists. The user can request any language, any tone,
# any goal, any content type, and any positive number of headlines.
SUGGESTED_CONTENT_TYPES = [
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
]

SUGGESTED_GOALS = [
    "Attract Attention",
    "Explain Clearly",
    "Increase Clicks",
    "Sound Professional",
    "Sound Creative",
    "Improve SEO",
    "Create Curiosity",
    "Sell / Convert",
    "Summarize Content",
]

SUGGESTED_LANGUAGES = [
    "Auto Detect",
    "Arabic",
    "English",
    "French",
    "Chinese",
    "Russian",
]

SUGGESTED_TONES = [
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
]

SUGGESTED_HEADLINE_LENGTHS = [
    "Short",
    "Medium",
    "Long",
    "Auto",
]

SUGGESTED_COUNTS = [1, 2, 3, 4, 5, 10, 15, 20]

SUGGESTED_EXTRA_OPTIONS = [
    "Include SEO-friendly headlines",
    "Include curiosity-based headlines",
    "Include professional headlines",
    "Avoid clickbait",
    "Avoid exaggeration",
    "Generate headline + subheadline",
]

MAX_HEADLINE_COUNT = 100

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
    """
    Normalize the extractor JSON without enforcing fixed allowed values.
    Suggested values in the prompt are examples only; user-provided values are allowed.
    """
    clean = {
        "content": normalize_text(payload.get("content")),
        "content_type": normalize_text(payload.get("content_type")),
        "goal": normalize_text(payload.get("goal")),
        "language": normalize_text(payload.get("language")),
        "tone": normalize_text(payload.get("tone")),
        "number_of_headlines": payload.get("number_of_headlines"),
        "headline_length": normalize_text(payload.get("headline_length")),
        "extra_options": payload.get("extra_options") or [],
    }

    try:
        if clean["number_of_headlines"] is not None and clean["number_of_headlines"] != "":
            count = int(clean["number_of_headlines"])
            clean["number_of_headlines"] = count if 1 <= count <= MAX_HEADLINE_COUNT else None
        else:
            clean["number_of_headlines"] = None
    except (TypeError, ValueError):
        clean["number_of_headlines"] = None

    if not isinstance(clean["extra_options"], list):
        clean["extra_options"] = [clean["extra_options"]]

    options: list[str] = []
    for option in clean["extra_options"]:
        option = str(option).strip()
        if option and option not in options:
            options.append(option)
    clean["extra_options"] = options

    return clean



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
        "content_type": "ما نوع المحتوى؟ يمكنك كتابة أي نوع، مثل: Article, News, YouTube Video, Social Media Post, Product, General",
        "goal": "ما الهدف من العناوين؟ يمكنك كتابة أي هدف، مثل: Attract Attention, Improve SEO, Increase Clicks, Create Curiosity",
        "language": "ما اللغة المطلوبة؟ يمكنك كتابة أي لغة، مثل: Arabic, English, French, Spanish, Turkish",
        "tone": "ما النبرة المطلوبة؟ يمكنك كتابة أي نبرة، مثل: Professional, Powerful, Simple, Creative, Marketing",
        "number_of_headlines": "كم عدد العناوين المطلوبة؟ اكتب أي رقم، مثل: 1 أو 2 أو 4 أو 10",
        "headline_length": "ما طول العنوان؟ يمكنك كتابة أي وصف، مثل: Short, Medium, Long, Auto, very short",
    }

    labels_en = {
        "content": "What is the content, topic, product, or idea you want headlines for?",
        "content_type": "What is the content type? You can write any type, for example: Article, News, YouTube Video, Social Media Post, Product, General",
        "goal": "What is the goal? You can write any goal, for example: Attract Attention, Improve SEO, Increase Clicks, Create Curiosity",
        "language": "What language do you want? You can write any language, for example: Arabic, English, French, Spanish, Turkish",
        "tone": "What tone do you want? You can write any tone, for example: Professional, Powerful, Simple, Creative, Marketing",
        "number_of_headlines": "How many headlines do you want? Write any number, for example: 1, 2, 4, or 10",
        "headline_length": "What headline length do you want? You can write any description, for example: Short, Medium, Long, Auto, very short",
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

Suggested values only. These are examples, not restrictions:
content_type examples = ["Article", "News", "YouTube Video", "Social Media Post", "Ad", "Email Subject", "Landing Page", "Product", "Report", "Creative Text", "General"]
goal examples = ["Attract Attention", "Explain Clearly", "Increase Clicks", "Sound Professional", "Sound Creative", "Improve SEO", "Create Curiosity", "Sell / Convert", "Summarize Content"]
language examples = ["Auto Detect", "Arabic", "English", "French", "Chinese", "Russian", "Spanish", "Turkish"]. Accept any language requested by the user.
tone examples = ["Professional", "Powerful", "Simple", "Creative", "Emotional", "Luxury", "Bold", "Informative", "Journalistic", "Academic", "Marketing", "Neutral"]
number_of_headlines examples = [1, 2, 3, 4, 5, 10, 15, 20]. Accept any positive integer from the user.
headline_length examples = ["Short", "Medium", "Long", "Auto"]
extra_options examples = ["Include SEO-friendly headlines", "Include curiosity-based headlines", "Include professional headlines", "Avoid clickbait", "Avoid exaggeration", "Generate headline + subheadline"]

Arabic mapping rules:
- "مقال", "المقال", "لهذا المقال", "لهزا المقال" => content_type = "Article".
- "خبر" => content_type = "News".
- "عنوان" or "عناوين" means the user wants headline generation; do not put the word itself as content.
- "قوي", "ملفت", "جذاب" => tone = "Powerful" and goal = "Attract Attention".
- "احترافي" => tone = "Professional" and add "Include professional headlines".
- "سيو" or "SEO" => goal = "Improve SEO" and add "Include SEO-friendly headlines".
- Arabic text => language = "Arabic" unless another language is requested.
- If the user asks for a specific number of headlines/titles, use that exact number. Examples: "one headline" => 1, "عنوانين" => 2, "4 عناوين" => 4.
- Do not force number_of_headlines to 5, 10, 15, or 20.
- If the user requests any language not listed in examples, keep that language exactly.
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
        response_format=object_response_format("headline_extractor"),
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
            response_format=object_response_format("headline_extractor"),
        )

        try:
            extracted = normalize_extracted_payload(extract_json_object(repair_result.content))
            return extracted, repair_result, {
                "first_error": str(first_error),
                "first_raw": extractor_result.content,
                "repaired": True,
            }
        except Exception as second_error:
            raise ProviderOutputError(
                "Extractor failed to return valid JSON after retry. "
                f"First error: {first_error}. "
                f"Second error: {second_error}. "
                f"Extractor trace: {extractor_result.trace_id or 'n/a'} / "
                f"{extractor_result.generation_id or 'n/a'}. "
                f"Repair trace: {repair_result.trace_id or 'n/a'} / "
                f"{repair_result.generation_id or 'n/a'}."
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
                "extractor_trace": extractor_result.trace_metadata(),
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
            "extractor_trace": extractor_result.trace_metadata(),
            "repair": extractor_repair_debug,
            "generator_raw": generator_result.content,
            "generator_trace": generator_result.trace_metadata(),
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
