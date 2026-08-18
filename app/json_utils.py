from __future__ import annotations

import json
import re
from typing import Any


def _preview(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = text.replace("\x00", "")
    return text if len(text) <= limit else text[:limit] + "..."



def object_response_format(name: str) -> dict[str, Any]:
    """Return a strict JSON-Schema response format that requires a top-level object.

    Individual tool validators still enforce their detailed field contracts locally.
    This prevents valid JSON scalars/arrays (for example ``-1e308`` or ``[]``)
    from being accepted as a structured response by the upstream model.
    """
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "response")).strip("_")
    if not safe_name:
        safe_name = "response"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": safe_name[:64],
            "strict": True,
            "schema": {"type": "object"},
        },
    }


def extract_json_object(value: Any, *, label: str = "JSON") -> dict[str, Any]:
    """Parse a model response and require an object at the top level.

    Accepts clean JSON, JSON fenced in Markdown, or a JSON object embedded in a
    short textual prefix/suffix. Valid JSON scalars/arrays are rejected with an
    explicit type error instead of being treated as malformed JSON.
    """
    if isinstance(value, dict):
        return value
    if value is None:
        raise ValueError(f"{label} response is empty")
    if not isinstance(value, str):
        raise ValueError(
            f"{label} must be a JSON object; got {type(value).__name__}: {_preview(value)}"
        )

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} response is empty")

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        # Avoid a greedy {.*} regex, which can accidentally merge multiple JSON
        # objects. raw_decode parses one object from the first opening brace.
        start = cleaned.find("{")
        if start < 0:
            raise ValueError(
                f"{label} did not contain a JSON object: {_preview(value)}"
            ) from first_error
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as second_error:
            raise ValueError(
                f"{label} did not contain valid JSON: {_preview(value)}"
            ) from second_error

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{label} must be a JSON object; got {type(parsed).__name__}: {_preview(parsed)}"
        )
    return parsed
