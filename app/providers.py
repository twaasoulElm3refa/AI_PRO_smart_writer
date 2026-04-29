from typing import List, Optional
import httpx

from app.settings import get_settings
from app.schemas import ChatMessage
from app.tasks import MODEL_ROUTES


def _clean_ascii_header(value: str) -> str:
    return value.encode("ascii", "ignore").decode("ascii").strip()


async def send_messages_with_model(
    model_key: str,
    messages: List[ChatMessage],
    temperature_override: Optional[float] = None,
    max_tokens_override: Optional[int] = None,
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

    temperature = temperature_override if temperature_override is not None else route.get("temperature", settings.DEFAULT_TEMPERATURE)
    max_tokens = max_tokens_override if max_tokens_override is not None else route.get("max_tokens", settings.DEFAULT_MAX_TOKENS)

    url = f"{settings.OPENROUTER_BASE_URL}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [msg.model_dump() for msg in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}".strip(),
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": _clean_ascii_header(settings.APP_NAME),
    }

    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    return data["choices"][0]["message"]["content"].strip()