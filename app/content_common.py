from __future__ import annotations

from typing import Any

from app.providers import ProviderResult
from app.schemas import TokenUsage


def normalize_text(value: Any) -> str | None:
    """Return a stripped string, or None for an empty/missing value."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def merge_extra_options(old_options: list[str], new_options: list[str] | None) -> list[str]:
    """Merge option lists while preserving order and removing blanks/duplicates."""
    merged: list[str] = []
    for item in old_options + (new_options or []):
        text = str(item).strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def combine_usage(*items: ProviderResult | None) -> TokenUsage:
    """Aggregate token usage across one or more provider calls."""
    input_tokens = sum((item.input_tokens or 0) for item in items if item is not None)
    output_tokens = sum((item.output_tokens or 0) for item in items if item is not None)
    total_tokens = sum((item.total_tokens or 0) for item in items if item is not None)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
    )
