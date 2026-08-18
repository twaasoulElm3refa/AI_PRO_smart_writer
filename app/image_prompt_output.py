from __future__ import annotations

import json
import re
from typing import Any


from app.json_utils import object_response_format


IMAGE_PROMPT_RESPONSE_FORMAT: dict[str, Any] = object_response_format("image_prompt_output")

_NULL_LIKE_VALUES = {
    "",
    "null",
    "none",
    "undefined",
    "n/a",
    "na",
    "nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}

_NUMERIC_ONLY_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)

_CONVERSATIONAL_MARKERS = (
    "what i can do",
    "i can help you",
    "to get started",
    "please tell me",
    "provide more details",
    "i am your ai",
    "i'm your ai",
    "hello!",
    "here's what i can do",
)

_META_REQUEST_MARKERS = (
    "what can you do",
    "what you can do",
    "your capabilities",
    "who are you",
    "hello",
    "hi",
    "hey",
)



def strip_hidden_reasoning(value: Any) -> str:
    """Remove model reasoning wrappers and return safe visible text."""
    if value is None:
        return ""

    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False).strip()
        except (TypeError, ValueError):
            text = str(value).strip()

    if not text:
        return ""

    text = re.sub(
        r"(?is)<(?:think|analysis|reasoning)\b[^>]*>.*?</(?:think|analysis|reasoning)>",
        "",
        text,
    ).strip()

    # Some models omit the opening tag but leave a closing reasoning tag.
    for closing_tag in ("</think>", "</analysis>", "</reasoning>"):
        lowered = text.lower()
        if closing_tag in lowered:
            position = lowered.rfind(closing_tag)
            text = text[position + len(closing_tag):].strip()

    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    return text



def _extract_prompt_candidate(value: str) -> str:
    """Extract a prompt string from strict JSON or compatible provider output."""
    cleaned = strip_hidden_reasoning(value)
    if not cleaned:
        return ""

    try:
        decoded = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        decoded = None

    # Preferred output contract: {"prompt": "..."}
    if isinstance(decoded, dict):
        for key in ("prompt", "positive_prompt", "content", "text", "result"):
            candidate = decoded.get(key)
            if isinstance(candidate, str) and candidate.strip():
                cleaned = candidate.strip()
                break
        else:
            results = decoded.get("results")
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, str):
                    cleaned = first.strip()
                elif isinstance(first, dict):
                    for key in ("prompt", "positive_prompt", "content", "text"):
                        candidate = first.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            cleaned = candidate.strip()
                            break
    elif decoded is not None:
        # Numbers, booleans, arrays, and null are never valid image prompts.
        return ""

    # Remove labels if a provider ignores JSON mode and returns plain text.
    cleaned = re.sub(
        r"(?is)^\s*(?:here(?:'|’)s|here is)\s+(?:the\s+)?(?:final\s+)?(?:image\s+)?prompt\s*[:：-]?\s*",
        "",
        cleaned,
    ).strip()
    cleaned = re.sub(
        r"(?is)^\s*(?:final\s+)?(?:image\s+)?prompt\s*[:：-]\s*",
        "",
        cleaned,
    ).strip()

    # Keep only the positive prompt if the provider added separate sections.
    cleaned = re.split(
        r"(?im)^\s*(?:negative\s+prompt|suggested\s+aspect\s+ratio|aspect\s+ratio|notes?|explanation)\s*[:：-]",
        cleaned,
        maxsplit=1,
    )[0].strip()

    return cleaned.strip(' \n\t"')



def is_valid_image_prompt(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    text = value.strip()
    if len(text) < 40:
        return False

    lowered = text.lower()
    if lowered in _NULL_LIKE_VALUES:
        return False
    if _NUMERIC_ONLY_PATTERN.fullmatch(text):
        return False

    alphabetic_characters = sum(1 for character in text if character.isalpha())
    if alphabetic_characters < 12:
        return False

    if any(marker in lowered for marker in _CONVERSATIONAL_MARKERS):
        return False

    return True



def _meaningful_state_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _NULL_LIKE_VALUES | {"auto", "auto detect"}:
        return None
    return text



def build_fallback_image_prompt(
    source_message: str,
    state: dict[str, Any] | None,
) -> str:
    """Build a deterministic prompt when the provider output is invalid."""
    source = (source_message or "").strip()
    lowered = source.lower()

    if not source or any(marker in lowered for marker in _META_REQUEST_MARKERS):
        source = (
            "a sophisticated AI image-prompt specialist inside a futuristic creative studio, "
            "transforming rough visual ideas into polished cinematic concepts across floating "
            "composition boards, camera diagrams, lighting studies, color palettes, and detailed "
            "image previews"
        )

    state = state or {}
    details: list[str] = []

    style = _meaningful_state_value(state.get("style"))
    camera = _meaningful_state_value(state.get("camera"))
    lighting = _meaningful_state_value(state.get("lighting"))
    aspect_ratio = _meaningful_state_value(state.get("aspect_ratio"))
    negative_prompt = _meaningful_state_value(state.get("negative_prompt"))

    if style:
        details.append(f"visual style: {style}")
    if camera:
        details.append(f"camera and framing: {camera}")
    if lighting:
        details.append(f"lighting: {lighting}")
    if aspect_ratio:
        details.append(f"composition optimized for a {aspect_ratio} aspect ratio")

    text_policy = _meaningful_state_value(state.get("text_policy"))
    if text_policy and (
        "no text" in text_policy.lower()
        or "without text" in text_policy.lower()
        or "من غير كلام" in text_policy
        or "بدون نص" in text_policy
    ):
        details.append("no visible text, letters, captions, or watermark")

    face_policy = _meaningful_state_value(state.get("face_policy"))
    if face_policy and (
        "no face" in face_policy.lower()
        or "without face" in face_policy.lower()
        or "بدون وجوه" in face_policy
    ):
        details.append("no visible human faces")

    if negative_prompt:
        details.append(f"exclude {negative_prompt}")

    extra_options = state.get("extra_options")
    if isinstance(extra_options, list):
        details.extend(
            str(item).strip()
            for item in extra_options
            if str(item).strip()
        )

    detail_suffix = f", {', '.join(details)}" if details else ""

    return (
        f"Create a polished, visually compelling image of {source}, with a clear primary focal "
        "point, balanced professional composition, purposeful camera framing, realistic depth, "
        "controlled cinematic lighting, harmonious colors, detailed materials, natural shadows, "
        "clean edges, atmospheric dimensionality, and production-ready high-resolution quality"
        f"{detail_suffix}."
    )



def finalize_image_prompt_reply(
    raw_response: Any,
    *,
    source_message: str,
    state: dict[str, Any] | None,
) -> str:
    candidate = _extract_prompt_candidate(strip_hidden_reasoning(raw_response))
    if is_valid_image_prompt(candidate):
        return candidate

    return build_fallback_image_prompt(source_message, state)
