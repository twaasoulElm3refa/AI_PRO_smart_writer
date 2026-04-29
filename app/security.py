from fastapi import Header, HTTPException
from app.settings import get_settings


async def verify_internal_api_key(x_internal_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()

    if not settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY is not configured")

    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized internal request")