import re
from typing import Any


STATE_ENABLED_TASKS = {"writer", "summarizer", "paraphraser", "image_prompt_generator"}
NULL_LIKE_TEXT = {"null", "none", "undefined", "n/a", "na"}


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in NULL_LIKE_TEXT:
        return None
    return value


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    cleaned: list[str] = []
    for item in value:
        item = str(item).strip()
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def normalize_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on", "include", "with", "نعم", "ايوه", "أيوه", "اه", "أه"}:
        return True
    if text in {"false", "0", "no", "n", "off", "without", "لا", "بدون"}:
        return False
    return default


def normalize_count(value: Any, default: int = 1, maximum: int = 50) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(maximum, parsed))


def looks_arabic(text: str | None) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def default_task_state(task_key: str) -> dict[str, Any]:
    if task_key == "writer":
        return {
            "content": None,
            "content_type": "General Writing",
            "language": "Auto Detect",
            "tone": "Professional",
            "audience": "General Audience",
            "goal": "Write clear, useful content",
            "length": "Auto",
            "output_format": "Auto",
            "keywords": [],
            "extra_options": [],
            "last_output": None,
        }
    if task_key == "summarizer":
        return {
            "content": None,
            "language": "Auto Detect",
            "summary_type": "General Summary",
            "length": "Concise",
            "audience": "General Audience",
            "focus_points": [],
            "output_format": "Paragraphs",
            "include_bullets": False,
            "extra_options": [],
            "last_output": None,
        }
    if task_key == "paraphraser":
        return {
            "content": None,
            "language": "Auto Detect",
            "tone": "Natural",
            "rewrite_mode": "Improve clarity and flow",
            "change_level": "Medium",
            "results_count": 1,
            "extra_options": [],
            "last_output": None,
        }
    if task_key == "image_prompt_generator":
        return {
            "content": None,
            "language": "Auto Detect",
            "style": "Auto",
            "aspect_ratio": "Auto",
            "camera": "Auto",
            "lighting": "Auto",
            "negative_prompt": None,
            "text_policy": "No text unless requested",
            "face_policy": "Follow the user request",
            "results_count": 1,
            "extra_options": [],
            "last_output": None,
        }
    return {}


def merge_incoming_state(task_key: str, incoming_state: dict[str, Any] | None) -> dict[str, Any]:
    data = default_task_state(task_key)
    if task_key not in STATE_ENABLED_TASKS:
        return incoming_state or {}
    if incoming_state:
        for key, value in incoming_state.items():
            if key not in data:
                continue
            if key in {"extra_options", "keywords", "focus_points"}:
                # An explicit empty list means clear the previous value.
                data[key] = normalize_list(value)
            elif key in {"include_bullets"}:
                data[key] = normalize_bool(value, data.get(key))
            elif key in {"results_count"}:
                data[key] = normalize_count(value, default=data.get(key) or 1, maximum=20)
            elif isinstance(value, str):
                cleaned = normalize_text(value)
                data[key] = cleaned if cleaned is not None else data[key]
            elif value is not None:
                data[key] = value
    return data


def is_follow_up_edit(message: str, state: dict[str, Any]) -> bool:
    if not normalize_text(state.get("last_output")):
        return False
    text = (message or "").strip().lower()
    edit_patterns = [
        r"\b(make|turn|change|rewrite|shorten|expand|translate|improve|humanize|summarize|convert|edit)\b",
        r"\b(shorter|longer|more formal|more casual|professional|friendly|arabic|english)\b",
        r"\b(خلي|خليه|اجعله|اختصر|لخص|طول|زود|حسن|عدّل|عدل|ترجم|بالعربي|بالانجليزي|رسمي|أقصر|اطول|أطول)\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in edit_patterns)


def detect_language(message: str, current_language: str | None = None) -> str | None:
    text = message or ""
    if re.search(r"\b(arabic|عربي|العربية|بالعربي)\b", text, flags=re.IGNORECASE):
        return "Arabic"
    if re.search(r"\b(english|انجليزي|إنجليزي|بالانجليزي)\b", text, flags=re.IGNORECASE):
        return "English"
    if looks_arabic(text):
        return "Arabic"
    return current_language


def detect_length(message: str, current_length: str | None = None) -> str | None:
    text = (message or "").lower()
    if re.search(r"\b(short|brief|concise|مختصر|قصير|أقصر|اقصر)\b", text):
        return "Short"
    if re.search(r"\b(long|detailed|تفصيلي|مفصل|طويل|أطول|اطول)\b", text):
        return "Long"
    if re.search(r"\bmedium|متوسط\b", text):
        return "Medium"
    words = re.search(r"(\d{2,5})\s*(?:words|كلمة|كلمات)", text)
    if words:
        return f"{words.group(1)} words"
    return current_length


def detect_output_format(message: str, current_format: str | None = None) -> str | None:
    text = (message or "").lower()
    if re.search(r"\b(bullets|bullet points|نقاط|قائمة)\b", text):
        return "Bullet points"
    if re.search(r"\b(table|جدول)\b", text):
        return "Table"
    if re.search(r"\b(json)\b", text):
        return "JSON"
    if re.search(r"\b(paragraph|paragraphs|فقرات|فقرة)\b", text):
        return "Paragraphs"
    return current_format


def detect_writer_content_type(message: str, current_type: str | None = None) -> str | None:
    text = (message or "").lower()
    mapping = [
        (r"\b(article|blog|مقال|مقالة)\b", "Article"),
        (r"\b(news|خبر|خبري)\b", "News"),
        (r"\b(linkedin|facebook|instagram|twitter|x post|social|بوست|منشور|تغريدة)\b", "Social Media Post"),
        (r"\b(email|ايميل|إيميل|بريد)\b", "Email"),
        (r"\b(report|تقرير)\b", "Report"),
        (r"\b(ad|advertisement|اعلان|إعلان)\b", "Ad Copy"),
        (r"\b(script|سكريبت|سيناريو)\b", "Script"),
        (r"\b(caption|كابشن)\b", "Caption"),
        (r"\b(story|قصة)\b", "Story"),
        (r"\b(seo|سيو)\b", "SEO Content"),
    ]
    for pattern, label in mapping:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return current_type


def detect_tone(message: str, current_tone: str | None = None) -> str | None:
    text = (message or "").lower()
    mapping = [
        (r"\b(professional|احترافي|مهني|رسمي)\b", "Professional"),
        (r"\b(friendly|ودود)\b", "Friendly"),
        (r"\b(marketing|تسويقي)\b", "Marketing"),
        (r"\b(journalistic|صحفي|خبري)\b", "Journalistic"),
        (r"\b(persuasive|مقنع|إقناعي)\b", "Persuasive"),
        (r"\b(simple|بسيط)\b", "Simple"),
        (r"\b(strong|powerful|قوي)\b", "Powerful"),
        (r"\b(natural|طبيعي|بشري)\b", "Natural"),
    ]
    for pattern, label in mapping:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return current_tone


def detect_paraphrase_mode(message: str, current_mode: str | None = None) -> str | None:
    text = (message or "").lower()
    if re.search(r"\b(shorten|مختصر|اختصر|أقصر|اقصر)\b", text):
        return "Shorten"
    if re.search(r"\b(expand|longer|طول|زود|أطول|اطول)\b", text):
        return "Expand"
    if re.search(r"\b(proofread|grammar|صحح|تدقيق)\b", text):
        return "Proofread"
    if re.search(r"\b(humanize|بشري|طبيعي)\b", text):
        return "Humanize"
    if re.search(r"\b(rewrite|paraphrase|صياغة|أعد صياغة|اعادة صياغة)\b", text):
        return "Paraphrase"
    return current_mode


def update_task_state(task_key: str, incoming_state: dict[str, Any] | None, user_message: str) -> dict[str, Any] | None:
    if task_key not in STATE_ENABLED_TASKS:
        return incoming_state

    state = merge_incoming_state(task_key, incoming_state)
    message = (user_message or "").strip()
    follow_up = is_follow_up_edit(message, state)

    language = detect_language(message, state.get("language"))
    if language:
        state["language"] = language

    if task_key == "writer":
        state["content_type"] = detect_writer_content_type(message, state.get("content_type"))
        state["tone"] = detect_tone(message, state.get("tone"))
        state["length"] = detect_length(message, state.get("length"))
        state["output_format"] = detect_output_format(message, state.get("output_format"))
        if re.search(r"\b(seo|سيو)\b", message, flags=re.IGNORECASE):
            state["goal"] = "SEO-friendly writing"
        elif re.search(r"\b(answer|جاوب|ما هو|مين|what|who|when|where|why|how)\b", message, flags=re.IGNORECASE):
            state["goal"] = "Answer the user question clearly"
        if not follow_up:
            state["content"] = message
        else:
            options = normalize_list(state.get("extra_options"))
            if message and message not in options:
                options.append(message)
            state["extra_options"] = options

    elif task_key == "summarizer":
        state["length"] = detect_length(message, state.get("length"))
        state["output_format"] = detect_output_format(message, state.get("output_format"))
        state["include_bullets"] = state.get("output_format") == "Bullet points" or state.get("include_bullets") is True
        if re.search(r"\b(detailed|تفصيلي|مفصل)\b", message, flags=re.IGNORECASE):
            state["summary_type"] = "Detailed Summary"
        elif re.search(r"\b(short|brief|مختصر|قصير)\b", message, flags=re.IGNORECASE):
            state["summary_type"] = "Brief Summary"
        if not follow_up:
            state["content"] = message
        else:
            options = normalize_list(state.get("extra_options"))
            if message and message not in options:
                options.append(message)
            state["extra_options"] = options

    elif task_key == "paraphraser":
        state["tone"] = detect_tone(message, state.get("tone"))
        state["rewrite_mode"] = detect_paraphrase_mode(message, state.get("rewrite_mode"))
        length = detect_length(message, None)
        if length == "Short":
            state["change_level"] = "Light and concise"
        elif length == "Long":
            state["change_level"] = "Expanded"
        count_match = re.search(r"(?:give me|generate|هات|اعمل|اكتب)?\s*(\d{1,2})", message, flags=re.IGNORECASE)
        if count_match:
            state["results_count"] = normalize_count(count_match.group(1), default=state.get("results_count") or 1, maximum=10)
        if not follow_up:
            state["content"] = message
        else:
            options = normalize_list(state.get("extra_options"))
            if message and message not in options:
                options.append(message)
            state["extra_options"] = options

    elif task_key == "image_prompt_generator":
        if not follow_up:
            state["content"] = message
        else:
            options = normalize_list(state.get("extra_options"))
            if message and message not in options:
                options.append(message)
            state["extra_options"] = options

        ratio = re.search(r"\b(1:1|4:5|3:4|2:3|3:2|16:9|9:16|21:9)\b", message)
        if ratio:
            state["aspect_ratio"] = ratio.group(1)
        if re.search(r"\b(no text|without text|بدون نص|من غير كلام|لا نص)\b", message, flags=re.IGNORECASE):
            state["text_policy"] = "No text"
        if re.search(r"\b(no faces|without faces|بدون وجوه|لا وجوه)\b", message, flags=re.IGNORECASE):
            state["face_policy"] = "No visible faces"

    return state


def finalize_task_state(task_key: str, state: dict[str, Any] | None, reply: str) -> dict[str, Any] | None:
    if task_key not in STATE_ENABLED_TASKS:
        return state
    final_state = merge_incoming_state(task_key, state)
    final_state["last_output"] = (reply or "").strip() or final_state.get("last_output")
    return final_state


def build_task_state_context(task_key: str, state: dict[str, Any] | None) -> str:
    if task_key not in STATE_ENABLED_TASKS or not state:
        return ""

    labels = {
        "writer": "Writer state",
        "summarizer": "Summarizer state",
        "paraphraser": "Paraphraser state",
        "image_prompt_generator": "Image prompt generator state",
    }
    return f"""
Saved {labels.get(task_key, 'task state')}:
{state}

State usage rules:
- Use this state as additional structured context for the current request.
- If last_output exists and the current user request asks for an edit, revise last_output instead of starting from zero.
- If content exists, treat it as the main source/topic/text for this task.
- Follow all populated state fields such as language, tone, style, aspect_ratio, camera, lighting, output_format, and extra_options.
""".strip()
