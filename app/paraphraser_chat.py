import json
import re
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.crud import get_existing_conversation_for_task, get_recent_messages
from app.content_common import combine_usage, merge_extra_options, normalize_text
from app.errors import ProviderOutputError
from app.json_utils import extract_json_object, object_response_format
from app.providers import send_messages_with_model
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
# Examples only. The backend accepts ANY language requested by the user.
SUGGESTED_LANGUAGES = [
    "Auto Detect",
    "Arabic",
    "English",
    "French",
    "Spanish",
    "German",
    "Italian",
    "Turkish",
    "Russian",
    "Chinese",
    "Japanese",
    "Korean",
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


def normalize_paraphraser_state(state: Optional[ParaphraserState]) -> ParaphraserState:
    """Convert null/partial frontend state into a complete internal state.

    This allows the frontend to always send the same schema, even on the first
    request:
    {
        "content": null,
        "language": null,
        "tone": null,
        "rewrite_mode": null,
        "change_level": null,
        "results_count": null,
        "extra_options": []
    }

    Null values are treated as missing values and replaced by backend defaults.
    """
    defaults = default_paraphraser_state()

    if state is None:
        return defaults

    return ParaphraserState(
        content=state.content or defaults.content,
        language=state.language or defaults.language,
        tone=state.tone or defaults.tone,
        rewrite_mode=state.rewrite_mode or defaults.rewrite_mode,
        change_level=state.change_level or defaults.change_level,
        results_count=state.results_count or defaults.results_count,
        extra_options=state.extra_options or defaults.extra_options,
    )




def finalize_paraphraser_state_for_response(
    state: Optional[ParaphraserState],
    content: Optional[str] = None,
) -> ParaphraserState:
    """Return a complete state object for frontend/Laravel.

    The frontend may send nullable state fields. Internally we normalize them
    before generation, and we also normalize them again before returning. This
    prevents response states like tone=null, rewrite_mode=null, results_count=null.

    If content is provided, it becomes the latest assistant output stored in
    state.content, so the next follow-up edit works on the text the user saw.
    """
    normalized = normalize_paraphraser_state(state)

    if content is not None:
        normalized.content = normalize_text(content)

    return normalize_paraphraser_state(normalized)


def is_empty_paraphraser_state(state: Optional[ParaphraserState]) -> bool:
    """True when the client sent no usable state.

    Supports both:
    - state: null
    - state object with all values null/empty
    """
    if state is None:
        return True

    return not any(
        [
            state.content,
            state.language,
            state.tone,
            state.rewrite_mode,
            state.change_level,
            state.results_count,
            state.extra_options,
        ]
    )


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




def normalize_for_intent(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[\s\u200f\u200e]+", " ", value)
    value = value.strip(" .,!؟?؛:،\"'“”")
    return value


def clean_language_name(value: str | None) -> str | None:
    """Normalize a language name without restricting it to a fixed list."""
    if not value:
        return None

    value = str(value).strip()
    value = re.sub(r"^[\s:：,،؛.!?؟\-]+|[\s:：,،؛.!?؟\-]+$", "", value)
    value = value.strip("'\"“”()[]{}")
    value = re.sub(r"\s+", " ", value)

    # Remove Arabic definite article when users write things like: بالفرنسية / اللغة الفرنسية.
    value = re.sub(r"^(?:ال)", "", value)
    value = value.strip()
    if not value:
        return None

    aliases = {
        # English names
        "english": "English",
        "arabic": "Arabic",
        "french": "French",
        "spanish": "Spanish",
        "german": "German",
        "italian": "Italian",
        "turkish": "Turkish",
        "russian": "Russian",
        "chinese": "Chinese",
        "japanese": "Japanese",
        "korean": "Korean",
        "portuguese": "Portuguese",
        "hindi": "Hindi",
        "urdu": "Urdu",
        "persian": "Persian",
        "farsi": "Persian",
        "dutch": "Dutch",
        "greek": "Greek",
        "hebrew": "Hebrew",
        "swedish": "Swedish",
        "norwegian": "Norwegian",
        "danish": "Danish",
        "finnish": "Finnish",
        "polish": "Polish",
        "indonesian": "Indonesian",
        "malay": "Malay",
        "thai": "Thai",
        "vietnamese": "Vietnamese",
        "swahili": "Swahili",
        # Arabic names and adjectives
        "انجليزي": "English",
        "إنجليزي": "English",
        "انجليزية": "English",
        "إنجليزية": "English",
        "انجليزيه": "English",
        "إنجليزيه": "English",
        "عربي": "Arabic",
        "عربية": "Arabic",
        "عربيه": "Arabic",
        "فرنسي": "French",
        "فرنسية": "French",
        "فرنسيه": "French",
        "فرنساوي": "French",
        "اسباني": "Spanish",
        "إسباني": "Spanish",
        "اسبانية": "Spanish",
        "إسبانية": "Spanish",
        "اسبانيه": "Spanish",
        "إسبانيه": "Spanish",
        "الماني": "German",
        "ألماني": "German",
        "المانية": "German",
        "ألمانية": "German",
        "المانيه": "German",
        "ألمانيه": "German",
        "ايطالي": "Italian",
        "إيطالي": "Italian",
        "ايطالية": "Italian",
        "إيطالية": "Italian",
        "ايطاليه": "Italian",
        "إيطاليه": "Italian",
        "تركي": "Turkish",
        "تركية": "Turkish",
        "تركيه": "Turkish",
        "روسي": "Russian",
        "روسية": "Russian",
        "روسيه": "Russian",
        "صيني": "Chinese",
        "صينية": "Chinese",
        "صينيه": "Chinese",
        "ياباني": "Japanese",
        "يابانية": "Japanese",
        "يابانيه": "Japanese",
        "كوري": "Korean",
        "كورية": "Korean",
        "كوريه": "Korean",
        "برتغالي": "Portuguese",
        "برتغالية": "Portuguese",
        "برتغاليه": "Portuguese",
        "هندي": "Hindi",
        "هندية": "Hindi",
        "هنديه": "Hindi",
        "اردو": "Urdu",
        "أردو": "Urdu",
        "فارسي": "Persian",
        "فارسية": "Persian",
        "فارسيه": "Persian",
        "هولندي": "Dutch",
        "هولندية": "Dutch",
        "هولنديه": "Dutch",
        "يوناني": "Greek",
        "يونانية": "Greek",
        "يونانيه": "Greek",
        "عبري": "Hebrew",
        "عبرية": "Hebrew",
        "عبريه": "Hebrew",
        "سويدي": "Swedish",
        "سويدية": "Swedish",
        "سويديه": "Swedish",
        "نرويجي": "Norwegian",
        "نرويجية": "Norwegian",
        "نرويجيه": "Norwegian",
        "دنماركي": "Danish",
        "دنماركية": "Danish",
        "دنماركيه": "Danish",
        "فنلندي": "Finnish",
        "فنلندية": "Finnish",
        "فنلنديه": "Finnish",
        "بولندي": "Polish",
        "بولندية": "Polish",
        "بولنديه": "Polish",
        "اندونيسي": "Indonesian",
        "إندونيسي": "Indonesian",
        "اندونيسية": "Indonesian",
        "إندونيسية": "Indonesian",
        "تايلاندي": "Thai",
        "تايلاندية": "Thai",
        "فيتنامي": "Vietnamese",
        "فيتنامية": "Vietnamese",
        "سواحلي": "Swahili",
        "سواحلية": "Swahili",
    }

    key = value.lower()
    return aliases.get(key) or aliases.get(value) or value.title()


def detect_requested_language(user_message: str) -> str | None:
    """Detect any requested output language from short follow-up/control messages.

    This is intentionally dynamic: it is not limited to Arabic/English.
    It catches patterns like:
    - continue in French
    - rewrite in Spanish
    - in German
    - French
    - خليه بالفرنسية
    - كمل باللغة السواحلية
    """
    raw = (user_message or "").strip()
    text = normalize_for_intent(raw)
    if not text:
        return None

    # English control patterns: continue in French / reply in Spanish / write with German.
    english_patterns = [
        r"\b(?:continue|reply|respond|answer|write|rewrite|paraphrase|make it|make this|use|switch)\s+(?:in|to|with)\s+([a-z][a-z\s\-]{1,40})\b",
        r"\b(?:in|to)\s+([a-z][a-z\s\-]{1,40})\b",
    ]
    for pattern in english_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1)
            candidate = re.split(r"\b(?:please|pls|now|next|style|language|tone)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
            lang = clean_language_name(candidate)
            if lang:
                return lang

    # Single-word language command: French / German / Spanish.
    if re.fullmatch(r"[a-z][a-z\-]{2,30}", text, flags=re.IGNORECASE):
        excluded = {"hi", "hey", "hello", "continue", "rewrite", "paraphrase", "shorter", "longer", "humanize", "professional", "simple"}
        if text.lower() not in excluded:
            return clean_language_name(text)

    # Arabic patterns with explicit language words.
    arabic_patterns = [
        r"(?:باللغة|بلغة)\s+([\u0600-\u06FFa-zA-Z][\u0600-\u06FFa-zA-Z\s\-]{1,40})",
        r"(?:خليه|خليها|اجعله|اجعلها|اكتبه|اكتبها|اكتب|كمل|واصل|رد|جاوب|صيغه|صيغها)\s+(?:ب|بالـ|بال|بـ)\s*([\u0600-\u06FFa-zA-Z][\u0600-\u06FFa-zA-Z\s\-]{1,40})",
        r"(?:^|\s)(?:ب|بالـ|بال|بـ)\s*([\u0600-\u06FF]{3,30})(?:\s|$)",
    ]
    for pattern in arabic_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1)
            # Avoid treating style words as languages.
            if candidate.strip() in {"احترافي", "بسيط", "اقصر", "أقصر", "اطول", "أطول", "بشري", "طبيعي"}:
                continue
            lang = clean_language_name(candidate)
            if lang:
                return lang

    return None

def is_greeting_only(user_message: str) -> bool:
    text = normalize_for_intent(user_message)
    greetings = {
        "hi",
        "hello",
        "hey",
        "مرحبا",
        "مرحباً",
        "اهلا",
        "أهلا",
        "اهلاً",
        "أهلاً",
        "هاي",
        "السلام عليكم",
        "صباح الخير",
        "مساء الخير",
    }
    return text in greetings


def is_small_control_message_without_content(user_message: str) -> bool:
    """Messages that configure the next answer but do not contain text to paraphrase."""
    text = normalize_for_intent(user_message)
    if not text:
        return True

    short_control_phrases = {
        "continue",
        "كمل",
        "واصل",
        "تابع",
    }
    if text in short_control_phrases:
        return True

    # Any short language-only/control message should update state.language dynamically.
    # Examples: continue in French, in German, Spanish, خليه بالفرنسية, كمل باللغة السواحلية.
    if len(text.split()) <= 6 and detect_requested_language(user_message):
        return True

    return False

def build_smart_missing_content_message(user_message: str, state: ParaphraserState) -> str:
    requested_language = detect_requested_language(user_message)
    user_text = normalize_for_intent(user_message)

    if requested_language:
        state.language = requested_language

    # Response language only for the helper/question message, not the paraphrased output.
    # The actual output language is saved dynamically in state.language and can be ANY language.
    wants_english_reply = bool(re.search(r"[A-Za-z]", user_message)) and not re.search(r"[\u0600-\u06FF]", user_message)

    if is_greeting_only(user_message):
        if wants_english_reply:
            return "Hi! Send me the text you want to paraphrase, and tell me any language or style you want."
        return "أهلاً بك! أرسل النص الذي تريد إعادة صياغته، ويمكنك تحديد أي لغة أو أسلوب تريده."

    if user_text in {"continue", "كمل", "واصل", "تابع"}:
        if wants_english_reply:
            return "Sure — I can continue, but there is no saved text yet. Send the text you want to paraphrase first."
        return "تمام، أقدر أكمل، لكن لا يوجد نص محفوظ بعد. أرسل النص الذي تريد إعادة صياغته أولًا."

    if requested_language:
        if wants_english_reply:
            return f"Sure — I’ll use {requested_language}. Send the text you want to paraphrase."
        return f"تمام — سأستخدم لغة: {requested_language}. أرسل النص الذي تريد إعادة صياغته."

    return get_question_message(state)

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
        # instruction: 'long text'
        r"[:：]\s*['\"“”‘’`](.{40,})['\"“”‘’`]\s*$",
        # instruction before quoted long text, even without colon
        r"['\"“”‘’`](.{40,})['\"“”‘’`]\s*$",
    ]
    for pattern in quote_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            content = normalize_arabic_text(match.group(1).strip().strip("'\"“”‘’`"))
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

    # Users often paste content and end with a weak/incomplete command such as:
    # "... اعادة", "... اعمل", "... rewrite". Treat the part before the
    # trailing command as the content when it is long enough.
    weak_trailing_commands = [
        "اعمل اعادة صياغة",
        "اعمل إعادة صياغة",
        "اعادة صياغة",
        "إعادة صياغة",
        "اعد صياغة",
        "أعد صياغة",
        "اعادة",
        "إعادة",
        "اعد",
        "أعد",
        "اعمل",
        "rewrite",
        "paraphrase",
    ]
    stripped_quotes_text = text.strip().strip("'\"“”‘’` ")
    for command in sorted(weak_trailing_commands, key=len, reverse=True):
        pattern = rf"\s+{re.escape(command)}\s*$"
        cleaned = re.sub(pattern, "", stripped_quotes_text, flags=re.IGNORECASE).strip()
        if cleaned != stripped_quotes_text and len(cleaned) >= 40:
            content = normalize_arabic_text(cleaned.strip().strip("'\"“”‘’` "))
            return content, "[USER_SENT_CONTENT_WITH_TRAILING_COMMAND_REMOVED]"

    # If the user pasted a long text only, treat the whole message as content.
    # Follow-up edit messages are usually short, so they will not hit this branch.
    if len(stripped_quotes_text) >= 120:
        return normalize_arabic_text(stripped_quotes_text), "[USER_SENT_CONTENT_ONLY]"

    return None, text




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
- "English", "in English", "continue in English", "reply in English" => language = "English".
- "Arabic", "in Arabic", "continue in Arabic", "reply in Arabic" => language = "Arabic".
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
        response_format=object_response_format("paraphraser_extractor"),
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
            response_format=object_response_format("paraphraser_extractor_repair"),
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


def calculate_dynamic_max_tokens(content: str, results_count: int, rewrite_mode: str | None, target_words: int | None = None) -> int:
    """Estimate enough output tokens for Arabic/long text plus JSON wrapping.

    The previous word-count-only estimate was too low for Arabic and caused the
    model to stop in the middle of {"results":[{"text":"..."}], which then made
    results[0].text contain a broken JSON string instead of direct text.

    If the user requests a target length such as "250 words", include that in
    the estimate. Otherwise the model may have enough tokens for a normal
    paraphrase, but not enough for expansion.
    """
    settings = get_settings()
    content = content or ""
    results_count = max(1, int(results_count or 1))

    word_count = len(content.split())
    char_count = len(content)

    # Arabic tokenization can be much more expensive than English word counts.
    base_tokens = max(word_count * 4.0, char_count / 2.0)

    if target_words:
        # A rough cross-language estimate. Arabic/JSON escaping often needs more
        # tokens than plain English word counts.
        base_tokens = max(base_tokens, float(target_words) * 3.2)

    multiplier = 1.25
    if rewrite_mode and rewrite_mode.lower() in {"shorter", "اختصار", "مختصر"}:
        multiplier = 0.8
    elif rewrite_mode and rewrite_mode.lower() in {"longer", "expand", "أطول", "اطول"}:
        multiplier = 1.8

    estimated = int((base_tokens * multiplier * results_count) + 500)
    estimated = max(settings.PARAPHRASER_MIN_OUTPUT_TOKENS, estimated)
    return min(settings.PARAPHRASER_MAX_OUTPUT_TOKENS, estimated)


def normalize_digits(value: str) -> str:
    """Convert Arabic/Persian numerals to ASCII so regex length parsing works."""
    translation = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return (value or "").translate(translation)


def extract_target_word_count(user_message: str) -> int | None:
    """Extract requests like '250 كلمة تقريبا' or 'about 250 words'."""
    text = normalize_digits(user_message or "")

    patterns = [
        r"(?:حوالي|حوالى|تقريبا|تقريبًا|about|around|approximately)\s*(\d{2,5})\s*(?:كلمة|كلمات|word|words)",
        r"(\d{2,5})\s*(?:كلمة|كلمات|word|words)\s*(?:تقريبا|تقريبًا|about|around|approximately)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        # Protect cost/performance and ignore tiny accidental numbers.
        if 20 <= value <= 3000:
            return value
    return None


def user_requested_longer(user_message: str) -> bool:
    text = normalize_for_intent(user_message or "")
    return bool(re.search(r"(?:اجعله|خليه|خليها|make it|make this).*(?:اطول|أطول|طويل|longer|expand)", text, flags=re.IGNORECASE)) or bool(
        re.search(r"(?:اطول|أطول|وسع|توسيع|longer|expand)", text, flags=re.IGNORECASE)
    )


def count_words_for_length_check(text: str) -> int:
    # Handles Arabic and Latin words better than split() with punctuation.
    return len(re.findall(r"[\w\u0600-\u06FF]+", text or "", flags=re.UNICODE))


def results_meet_target_length(results: list[ParaphraserResultItem], target_words: int | None) -> bool:
    if not target_words or not results:
        return True
    # For "about N words", accept 75% as a minimum. This avoids unnecessary
    # retries while still catching outputs that ignored the user's instruction.
    min_words = max(1, int(target_words * 0.75))
    return all(count_words_for_length_check(item.text) >= min_words for item in results)


def detect_instruction_intent(user_message: str | None) -> dict[str, Any]:
    """Detect broad edit/enhancement requests from the latest user message.

    The extractor only updates small settings. This function preserves the real
    editing goal so follow-up messages like:
    - اجعله أقوى
    - أضف أمثلة
    - حسّن البداية
    - خليه تسويقي
    - make it more persuasive
    are treated as the primary task, not reduced to a generic paraphrase.
    """
    raw = user_message or ""
    text = normalize_for_intent(raw)

    def has(pattern: str) -> bool:
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    return {
        "has_latest_instruction": bool(text),
        "wants_additions": has(r"(?:اضف|أضف|زود|زوّد|ضيف|أدخل|ادخل|add|include|insert)"),
        "wants_enhancement": has(r"(?:حسن|حسّن|تحسين|طور|طوّر|enhance|improve|polish|upgrade|make it better)"),
        "wants_examples": has(r"(?:مثال|امثلة|أمثلة|examples?)"),
        "wants_structure": has(r"(?:نقاط|عناوين|فقرات|ترتيب|قسم|قسّم|bullets?|headings?|paragraphs?|structure)"),
        "wants_more_persuasive": has(r"(?:اقوى|أقوى|مقنع|تسويقي|جذاب|مؤثر|persuasive|marketing|catchy|stronger)"),
        "wants_humanize": has(r"(?:بشري|طبيعي|غير آلي|مش ذكاء اصطناعي|زكاء اصطناعي|human|humanize|natural|not ai)"),
        "wants_shorter": has(r"(?:اختصر|مختصر|اقصر|أقصر|shorter|summarize|concise)"),
        "wants_longer": user_requested_longer(raw),
    }


def build_latest_instruction_block(latest_instruction: str | None, target_words: int | None) -> str:
    intent = detect_instruction_intent(latest_instruction)
    lines: list[str] = []

    if latest_instruction:
        lines.append(
            "Latest User Instruction - PRIMARY TASK, do not ignore it:\n"
            f"{latest_instruction}"
        )
    else:
        lines.append("Latest User Instruction - PRIMARY TASK:\nApply the selected rewrite settings to the text.")

    if target_words:
        lines.append(
            f"Length Requirement:\nWrite about {target_words} words. This is a hard requirement; do not ignore it."
        )

    enabled_intents = [key for key, value in intent.items() if value and key != "has_latest_instruction"]
    if enabled_intents:
        lines.append(
            "Detected User Edit Goals:\n"
            + ", ".join(enabled_intents)
        )

    lines.append(
        "Instruction Priority Rules:\n"
        "- The latest user instruction is more important than saved rewrite_mode, tone, and extra_options.\n"
        "- Apply the requested edit to the current text, not just a generic paraphrase.\n"
        "- If the user asks to add, enhance, expand, restructure, humanize, simplify, or make it more persuasive, do that directly.\n"
        "- If saved options conflict with the latest user instruction, follow the latest user instruction.\n"
        "- Keep factual integrity: do not invent specific facts, numbers, names, sources, dates, or claims. Generic clarifying wording and non-factual examples are allowed only when the user asks to add/enhance/examples."
    )

    return "\n\n".join(lines)


def build_generator_user_prompt(
    state: ParaphraserState,
    latest_instruction: str | None = None,
    target_words: int | None = None,
) -> str:
    instruction_block = build_latest_instruction_block(latest_instruction, target_words)
    return f"""
Apply the latest user instruction to the text below and return exactly {state.results_count} version(s).

{instruction_block}

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
- Apply the latest user instruction as the main task.
- Preserve the original core meaning and factual integrity.
- Do not remove important details unless the user asks for shortening/summarizing.
- Follow requested edits exactly: add, enhance, expand, shorten, humanize, simplify, restructure, translate, change tone, add examples, or improve style when requested.
- If the user asks for a longer text or a target word count, expand naturally using the ideas already present in the input.
- If the user asks to add examples/details, you may add generic explanatory wording that supports the existing meaning, but do not invent specific facts, numbers, names, dates, sources, or events.
- If saved extra options conflict with the latest instruction, the latest instruction wins.
- Make the wording natural, clear, and ready to use.
- If multiple versions are requested, make them meaningfully different.
- Do not include reasoning, planning, explanations, labels, or the original text.
- Return valid JSON only using this exact shape:
{{
  "results": [
    {{"text": "rewritten version here"}}
  ]
}}
""".strip()


def build_generator_retry_prompt(
    state: ParaphraserState,
    invalid_output: str,
    latest_instruction: str | None = None,
    target_words: int | None = None,
    retry_reason: str | None = None,
) -> str:
    instruction_block = build_latest_instruction_block(latest_instruction, target_words)
    reason = retry_reason or "Your previous answer was invalid because it did not include a non-empty results[0].text value."
    return f"""
{reason}

Apply the latest user instruction to the same text again now.

{instruction_block}

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
- Follow the latest user instruction exactly, including requested edits and requested length/word count.
- If a target word count is requested, the rewritten text must be close to that word count.
- If the user asks to add/enhance/examples, apply that request while keeping factual integrity.
- Do not invent specific unsupported facts, numbers, names, dates, sources, or events.

Invalid previous output:
{invalid_output}

Return JSON only in this exact shape, with a real non-empty rewritten text string:
{{
  "results": [
    {{"text": "rewritten version here"}}
  ]
}}
""".strip()


def looks_like_json_payload(text: str) -> bool:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned.startswith("{") or cleaned.startswith("[") or cleaned.startswith('"{') or cleaned.startswith('"[')


def clean_result_text(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*(?:version|result|option)\s*\d+\s*[:.)-]\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\s*(?:النسخة|النتيجة|الخيار)\s*\d+\s*[:.)-]\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\s*\d+\s*[\).\-:]\s*", "", line)
    line = re.sub(r"^\s*[-•]\s*", "", line)
    return line.strip(" \t\r\n\"'“”")


def try_parse_json_value(value: str) -> Any | None:
    """Parse normal JSON or JSON that was returned as a quoted string."""
    cleaned = (value or "").strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    # Some providers return the JSON object as a JSON string:
    # "{\"results\":[{\"text\":\"...\"}]}"
    for _ in range(3):
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, str):
            next_value = parsed.strip()
            if next_value == cleaned:
                return parsed
            cleaned = next_value
            continue

        return parsed

    return None


def extract_jsonish_text_field(value: str) -> str | None:
    """Best-effort extraction when the model output was cut inside JSON."""
    candidates = [(value or "").strip()]
    relaxed = candidates[0].replace('\\"', '"').replace('\\n', '\n')
    if relaxed != candidates[0]:
        candidates.append(relaxed)

    for candidate in candidates:
        if not candidate:
            continue
        if not ("results" in candidate or '"text"' in candidate or "'text'" in candidate):
            continue

        match = re.search(
            r"[\"'](?:text|content|result)[\"']\s*:\s*[\"']((?:\\\\.|[^\"'\\\\])*)",
            candidate,
            flags=re.DOTALL,
        )
        if not match:
            continue

        raw = match.group(1).strip()
        if not raw:
            continue

        # Decode common JSON escapes without requiring a closing JSON object.
        try:
            decoded = json.loads(f'"{raw}"')
        except Exception:
            decoded = (
                raw.replace('\\"', '"')
                .replace("\\'", "'")
                .replace('\\n', '\n')
                .replace('\\r', '\r')
                .replace('\\t', '\t')
            )

        cleaned = clean_result_text(str(decoded))
        if cleaned:
            return cleaned

    return None


def coerce_result_text(value: Any, depth: int = 0) -> str:
    """Return the actual rewritten text, even if it is double-wrapped JSON."""
    if value is None or depth > 5:
        return ""

    if isinstance(value, dict):
        raw_results = value.get("results")
        if isinstance(raw_results, list):
            for item in raw_results:
                text = coerce_result_text(item, depth + 1)
                if text:
                    return text

        for key in ("text", "content", "result"):
            if key in value:
                text = coerce_result_text(value.get(key), depth + 1)
                if text:
                    return text

        return ""

    if isinstance(value, list):
        for item in value:
            text = coerce_result_text(item, depth + 1)
            if text:
                return text
        return ""

    text = str(value or "").strip()
    if not text:
        return ""

    parsed = try_parse_json_value(text)
    if parsed is not None and not isinstance(parsed, str):
        parsed_text = coerce_result_text(parsed, depth + 1)
        if parsed_text:
            return parsed_text

    # If parsed is a string, it may still be a JSON-ish string or normal text.
    if isinstance(parsed, str) and parsed.strip() != text:
        parsed_text = coerce_result_text(parsed, depth + 1)
        if parsed_text:
            return parsed_text

    jsonish = extract_jsonish_text_field(text)
    if jsonish:
        return jsonish

    return clean_result_text(text)


def build_result_items_from_any(value: Any, requested_count: int) -> list[ParaphraserResultItem]:
    requested_count = max(1, int(requested_count or 1))
    items: list[ParaphraserResultItem] = []

    def add_text(raw: Any):
        text = coerce_result_text(raw)
        if text:
            items.append(ParaphraserResultItem(id=len(items) + 1, text=text))

    if isinstance(value, dict) and isinstance(value.get("results"), list):
        for raw in value["results"]:
            add_text(raw)
            if len(items) >= requested_count:
                break
        return items

    if isinstance(value, list):
        for raw in value:
            add_text(raw)
            if len(items) >= requested_count:
                break
        return items

    add_text(value)
    return items[:requested_count]


def strip_generator_thinking(text: str) -> str:
    """Best-effort cleanup for models that leak reasoning despite JSON instructions."""
    text = (text or "").strip()
    if not text:
        return text

    # Prefer a JSON object if one exists anywhere in the response.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    # Remove common reasoning sections/prefixes from non-JSON fallback outputs.
    reasoning_markers = [
        r"(?is)^.*?let(?:'|’)s\s+(?:rewrite|craft|produce)[:.\s]*",
        r"(?is)^.*?we\s+need\s+to\s+rewrite.*?(?:let(?:'|’)s\s+(?:rewrite|craft|produce)[:.\s]*)",
        r"(?is)^.*?here\s+is\s+(?:the\s+)?(?:rewritten|paraphrased).*?:\s*",
        r"(?is)^.*?(?:النص\s+المعاد\s+صياغته|إعادة\s+الصياغة)\s*[:：]\s*",
    ]
    for pattern in reasoning_markers:
        cleaned = re.sub(pattern, "", text).strip()
        if cleaned != text and cleaned:
            return cleaned

    return text


def parse_paraphraser_results(text: str, requested_count: int) -> list[ParaphraserResultItem]:
    """Parse model output into clean result items.

    Supports:
    - Preferred JSON: {"results":[{"text":"..."}]}
    - Double-wrapped JSON strings inside text
    - Broken/truncated JSON where the useful text has already started
    - Plain text fallback
    """
    requested_count = max(1, int(requested_count or 1))
    text = strip_generator_thinking(text)

    parsed = try_parse_json_value(text)
    if parsed is not None:
        results = build_result_items_from_any(parsed, requested_count)
        if results:
            return results
        # Important: if the model returned JSON but with no usable text, do NOT
        # fall back to returning the raw JSON as the paraphrased text.
        return []

    # Existing helper can extract an object embedded in surrounding text.
    try:
        payload = extract_json_object(text)
        results = build_result_items_from_any(payload, requested_count)
        if results:
            return results
        return []
    except Exception:
        pass

    # Broken JSON fallback: {"results":[{"text":"actual text...
    jsonish = extract_jsonish_text_field(text)
    if jsonish:
        return [ParaphraserResultItem(id=1, text=jsonish)]

    # If it looks like JSON but no text field was recovered, treat it as invalid
    # so the caller can retry instead of showing {"results":[{}]} to the user.
    if looks_like_json_payload(text):
        return []

    # Plain text fallback.
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


def normalize_saved_role(role: str | None) -> str:
    return (role or "").strip().lower()


def is_non_content_assistant_message(text: str) -> bool:
    normalized = normalize_for_intent(text)
    non_content = {
        normalize_for_intent("تمت إعادة صياغة النص بنجاح."),
        normalize_for_intent("Text paraphrased successfully."),
        normalize_for_intent("ما النص الذي تريد إعادة صياغته؟"),
        normalize_for_intent("What text would you like me to paraphrase?"),
    }
    if normalized in non_content:
        return True
    if normalized.startswith(normalize_for_intent("تمت إعادة صياغة النص بنجاح")) and len(text) < 80:
        return True
    return False


def build_state_from_saved_assistant_content(content: str) -> ParaphraserState | None:
    """Recover the latest paraphraser state from a saved assistant message.

    Laravel may save either:
    - the full FastAPI JSON response,
    - only {"results":[{"text":"..."}]},
    - or just the assistant text.
    This function normalizes all of those into ParaphraserState.content.
    """
    raw = (content or "").strip()
    if not raw or is_non_content_assistant_message(raw):
        return None

    base = default_paraphraser_state().model_dump()
    parsed = try_parse_json_value(raw)

    if isinstance(parsed, dict):
        state_data = {}
        if isinstance(parsed.get("state"), dict):
            state_data = dict(parsed["state"])

        result_text = ""
        if "results" in parsed:
            result_text = coerce_result_text({"results": parsed.get("results")})
        if not result_text:
            result_text = coerce_result_text(parsed)
        if not result_text and state_data.get("content"):
            result_text = coerce_result_text(state_data.get("content"))

        if result_text and not is_non_content_assistant_message(result_text):
            state_data["content"] = result_text

        if state_data.get("content"):
            try:
                return ParaphraserState(**{**base, **state_data})
            except Exception:
                return ParaphraserState(content=state_data["content"])

    elif isinstance(parsed, list):
        result_text = coerce_result_text(parsed)
        if result_text and not is_non_content_assistant_message(result_text):
            return ParaphraserState(**{**base, "content": result_text})

    # Plain assistant text fallback.
    result_text = coerce_result_text(raw)
    if result_text and not is_non_content_assistant_message(result_text):
        return ParaphraserState(**{**base, "content": result_text})

    return None


def load_saved_paraphraser_state(db: Session, conversation_id: int) -> ParaphraserState | None:
    """Use the latest saved assistant reply when the client sends state=null."""
    for row in reversed(get_recent_messages(db, conversation_id, limit=30)):
        role = normalize_saved_role(getattr(row, "role", None))
        if role not in {"assistant", "ai"}:
            continue

        state = build_state_from_saved_assistant_content(getattr(row, "content", "") or "")
        if state and (state.content or "").strip():
            return state

    return None


async def run_paraphraser_chat(
    db: Session,
    req: ParaphraserChatRequest,
    request_id: str,
) -> ParaphraserChatResponse:
    settings = get_settings()

    conversation = get_existing_conversation_for_task(
        db=db,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
    )

    # State handling rules:
    # 1) If frontend sends a real state with content/settings, use it.
    # 2) If frontend sends state=null OR an empty state object with all fields null,
    #    try to recover the previous assistant output from DB.
    # 3) If nothing exists in DB, use backend defaults.
    incoming_state = req.state
    saved_state: ParaphraserState | None = None
    state_source = "request_state"

    if is_empty_paraphraser_state(incoming_state):
        saved_state = load_saved_paraphraser_state(db, conversation.id)
        if saved_state is not None:
            state_source = "saved_assistant_message"
        else:
            state_source = "default_state"

    state = normalize_paraphraser_state(saved_state or incoming_state)

    if len(req.user_message) > settings.MAX_USER_MESSAGE_LENGTH:
        raise ValueError(f"user_message exceeds max length of {settings.MAX_USER_MESSAGE_LENGTH}")

    detected_content, instruction_message = extract_paraphraser_content(req.user_message)

    # Smart no-content handling.
    # Greetings and setting-only messages like "hi" or "continue in english" should not be
    # treated as failed paraphrasing requests. If there is no saved content, answer naturally
    # and preserve any requested setting for the next turn without calling the extractor model.
    if not detected_content and not (state.content or "").strip() and (
        is_greeting_only(req.user_message) or is_small_control_message_without_content(req.user_message)
    ):
        smart_state = state.model_copy(deep=True)
        requested_language = detect_requested_language(req.user_message)
        if requested_language:
            smart_state.language = requested_language
        smart_state = finalize_paraphraser_state_for_response(smart_state)

        usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        cost = calculate_paraphraser_cost(input_tokens=0, output_tokens=0)

        debug = None
        if req.debug and settings.ENABLE_DEBUG_RESPONSE:
            debug = {
                "phase": "smart_no_content",
                "instruction_message": instruction_message,
                "detected_content": False,
                "requested_language": requested_language,
                "state_source": state_source,
                "state": smart_state.model_dump(),
            }

        return ParaphraserChatResponse(
            type="question",
            user_id=req.user_id,
            sub_tool_id=req.sub_tool_id,
            conversation_uuid=req.conversation_uuid,
            message=build_smart_missing_content_message(req.user_message, smart_state),
            state=smart_state,
            results=[],
            count=0,
            request_id=request_id,
            debug=debug,
            usage=usage,
            cost=cost,
        )

    extracted, extractor_result, extractor_repair_debug = await extract_paraphraser_updates_with_retry(
        state=state,
        user_message=instruction_message,
    )

    new_state = normalize_paraphraser_state(
        merge_paraphraser_state(state, extracted)
    )

    if detected_content:
        new_state.content = detected_content

    new_state = normalize_paraphraser_state(new_state)

    # The extractor only has a small settings schema, so exact instructions like
    # "اجعله اطول .. 250 كلمة تقريبا" can be lost unless we carry them into
    # the generator prompt explicitly. Parse them in Python and keep them as
    # custom constraints.
    target_word_count = extract_target_word_count(req.user_message)
    if target_word_count:
        target_option = f"Target length: about {target_word_count} words"
        if target_option not in new_state.extra_options:
            new_state.extra_options.append(target_option)

    if target_word_count or user_requested_longer(req.user_message):
        new_state.rewrite_mode = "Longer"

    if new_state.content and len(new_state.content) > settings.PARAPHRASER_MAX_CONTENT_CHARS:
        new_state.content = new_state.content[: settings.PARAPHRASER_MAX_CONTENT_CHARS].strip()

    if not is_ready_for_generation(new_state):
        new_state = finalize_paraphraser_state_for_response(new_state)

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
                "state_source": state_source,
                "state": new_state.model_dump(),
                "missing": get_missing_fields(new_state),
            }

        return ParaphraserChatResponse(
            type="question",
            user_id=req.user_id,
            sub_tool_id=req.sub_tool_id,
            conversation_uuid=req.conversation_uuid,
            message=build_smart_missing_content_message(req.user_message, new_state),
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
        target_words=target_word_count,
    )

    generator_messages = [
        ChatMessage(role="system", content=PARAPHRASER_GENERATOR_CHAT_PROMPT),
        ChatMessage(
            role="user",
            content=build_generator_user_prompt(
                new_state,
                latest_instruction=req.user_message,
                target_words=target_word_count,
            ),
        ),
    ]

    generator_result = await send_messages_with_model(
        model_key="paraphraser_fast",
        messages=generator_messages,
        temperature_override=0.35,
        max_tokens_override=max_tokens,
        enable_web_search=False,
        response_format=object_response_format("paraphraser_results"),
    )

    generator_results_for_usage = [generator_result]
    generator_retry_raw = None
    results = parse_paraphraser_results(generator_result.content, results_count)

    # Some models occasionally return valid JSON but with empty objects, e.g.
    # {"results":[{}]}. Also, some outputs ignore follow-up length instructions
    # such as "250 words". Retry once with an explicit correction prompt instead
    # of returning a bad/too-short result to the frontend.
    retry_reason = None
    if not results:
        retry_reason = "Your previous answer was invalid because it did not include a non-empty results[0].text value."
    elif not results_meet_target_length(results, target_word_count):
        actual_words = count_words_for_length_check(results[0].text)
        retry_reason = (
            f"Your previous answer was too short ({actual_words} words). "
            f"The user requested about {target_word_count} words. Rewrite again and meet the length requirement."
        )

    if retry_reason:
        retry_messages = [
            ChatMessage(role="system", content=PARAPHRASER_GENERATOR_CHAT_PROMPT),
            ChatMessage(
                role="user",
                content=build_generator_retry_prompt(
                    new_state,
                    generator_result.content,
                    latest_instruction=req.user_message,
                    target_words=target_word_count,
                    retry_reason=retry_reason,
                ),
            ),
        ]
        retry_result = await send_messages_with_model(
            model_key="paraphraser_fast",
            messages=retry_messages,
            temperature_override=0.2,
            max_tokens_override=max_tokens,
            enable_web_search=False,
            response_format=object_response_format("paraphraser_results_repair"),
        )
        generator_results_for_usage.append(retry_result)
        generator_retry_raw = retry_result.content
        results = parse_paraphraser_results(retry_result.content, results_count)

    if not results:
        new_state = finalize_paraphraser_state_for_response(new_state)

        usage = combine_usage(extractor_result, *generator_results_for_usage)
        cost = calculate_paraphraser_cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        wants_arabic_message = new_state.language == "Arabic" or bool(re.search(r"[\u0600-\u06FF]", req.user_message or ""))
        fail_message = (
            "لم يرجع النموذج نتيجة صالحة. حاول مرة أخرى أو أرسل النص داخل state.content."
            if wants_arabic_message
            else "The model returned an empty result. Try again or send the text inside state.content."
        )

        debug = None
        if req.debug and settings.ENABLE_DEBUG_RESPONSE:
            debug = {
                "phase": "generator_empty_result",
                "extracted": extracted,
                "extractor_raw": extractor_result.content,
                "instruction_message": instruction_message,
                "detected_content": bool(detected_content),
                "repair": extractor_repair_debug,
                "generator_raw": generator_result.content,
                "generator_retry_raw": generator_retry_raw,
                "state_source": state_source,
                "state": new_state.model_dump(),
                "max_tokens": max_tokens,
                "target_word_count": target_word_count,
            }

        return ParaphraserChatResponse(
            type="question",
            user_id=req.user_id,
            sub_tool_id=req.sub_tool_id,
            conversation_uuid=req.conversation_uuid,
            message=fail_message,
            state=new_state,
            results=[],
            count=0,
            request_id=request_id,
            debug=debug,
            usage=usage,
            cost=cost,
        )

    # Important for chat follow-ups:
    # after returning a paraphrased result, make the returned state.content point
    # to the latest assistant output, not only the original user input.
    # This lets the next message like "make it shorter", "continue in French",
    # or "خليه أبسط" edit the last result the user just saw.
    if results:
        new_state = finalize_paraphraser_state_for_response(
            new_state,
            content=results[0].text,
        )
    else:
        new_state = finalize_paraphraser_state_for_response(new_state)

    usage = combine_usage(extractor_result, *generator_results_for_usage)
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
            "generator_retry_raw": generator_retry_raw,
            "state_source": state_source,
            "state": new_state.model_dump(),
            "results_count": len(results),
            "max_tokens": max_tokens,
            "target_word_count": target_word_count,
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
