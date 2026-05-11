from typing import List, Optional
import httpx

from app.settings import get_settings
from app.schemas import ChatMessage
from app.tasks import MODEL_ROUTES

from dataclasses import dataclass

@dataclass
class ProviderResult:
    content: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw_usage: Optional[dict] = None
    

def _clean_ascii_header(value: str) -> str:
    cleaned = value.encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or "AI Tools API"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


async def send_messages_with_model(
    model_key: str,
    messages: List[ChatMessage],
    temperature_override: Optional[float] = None,
    max_tokens_override: Optional[int] = None,
    enable_web_search: bool = False,
    web_search_max_results: Optional[int] = None,
    web_search_max_total_results: Optional[int] = None,
) -> str:
    settings = get_settings()

    if model_key not in MODEL_ROUTES:
        raise ValueError(f"Unknown model_key: {model_key}")

    route = MODEL_ROUTES[model_key]
    provider = route["provider"]

    if provider != "openrouter":
        raise ValueError(f"Unsupported provider: {provider}")

    model_env_key = route["model_env_key"]
    model_name = getattr(settings, model_env_key, None)

    if not model_name:
        raise ValueError(f"Missing model name for env key: {model_env_key}")

    if not settings.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing")

    temperature = (
        float(temperature_override)
        if temperature_override is not None
        else float(route.get("temperature", settings.DEFAULT_TEMPERATURE))
    )

    max_tokens = (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(route.get("max_tokens", settings.DEFAULT_MAX_TOKENS))
    )

    url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [msg.model_dump() for msg in messages],
        "temperature": temperature,
        # OpenRouter supports max_completion_tokens and max_tokens. Use max_completion_tokens for newer models.
        "max_completion_tokens": max_tokens,
    }

    if enable_web_search:
        max_results = _clamp(
            web_search_max_results or settings.WEB_SEARCH_DEFAULT_MAX_RESULTS,
            1,
            settings.WEB_SEARCH_HARD_MAX_RESULTS,
        )
        max_total_results = _clamp(
            web_search_max_total_results or settings.WEB_SEARCH_DEFAULT_MAX_TOTAL_RESULTS,
            max_results,
            settings.WEB_SEARCH_HARD_MAX_TOTAL_RESULTS,
        )

        payload["tools"] = [
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "max_results": max_results,
                    "max_total_results": max_total_results,
                    "search_context_size": "medium",
                },
            }
        ]

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}".strip(),
        "Content-Type": "application/json",
        "HTTP-Referer": settings.SITE_URL,
        "X-Title": _clean_ascii_header(settings.APP_NAME),
    }

    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=payload, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise ValueError(f"OpenRouter error: {detail}") from exc
        data = response.json()

    try:
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
        )
        
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
        )
        
        total_tokens = usage.get("total_tokens")
        
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        
        return ProviderResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw_usage=usage,
        )
    
    except Exception as exc:
        raise ValueError(f"Unexpected OpenRouter response: {data}") from exc
