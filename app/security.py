import secrets

from fastapi import HTTPException, Request

from app.settings import get_settings


async def verify_internal_api_key(request: Request) -> None:
    settings = get_settings()
    configured_key = settings.INTERNAL_API_KEY
    if not configured_key:
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY is not configured")

    header_name = settings.INTERNAL_API_HEADER_NAME.strip() or "X-Internal-Api-Key"
    received_key = request.headers.get(header_name)
    if not received_key or not secrets.compare_digest(received_key, configured_key):
        raise HTTPException(status_code=401, detail="Unauthorized internal request")
