from __future__ import annotations

import json
import math

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.crud import get_ai_model, get_existing_conversation_for_model, list_ai_models
from app.media_tools import run_image_generator, run_speech_to_text, run_text_to_speech
from app.schemas import CostUsage, ImageGeneratorRequest, TextToSpeechRequest
from app.security import verify_internal_api_key
from app.settings import get_settings

from .catalog import build_catalog
from .model_catalog import serialize_ai_model, sync_openrouter_models
from .schemas import (
    GeneralAudioRequest,
    GeneralMediaRequest,
    GeneralTextRequest,
    GeneralToolResponse,
)
from .service import run_general_text


router = APIRouter(prefix="/tasks", tags=["general tools"])


def _error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400 if isinstance(exc, ValueError) else 502, detail=str(exc))


@router.get("/general-tools/catalog")
async def general_tools_catalog(
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    catalog = build_catalog()
    for tool in catalog["tools"]:
        rows, _ = list_ai_models(db, tool_key=tool["key"], page=1, per_page=100)
        tool["models"] = [serialize_ai_model(row) for row in rows]
    return catalog


@router.get("/general-tools/{tool_key}/models")
async def general_tool_models(
    tool_key: str,
    operation: str | None = None,
    provider: str | None = None,
    tier: str | None = None,
    search: str | None = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    rows, total = list_ai_models(
        db, tool_key=tool_key, operation=operation, provider=provider,
        tier=tier, search=search, page=page, per_page=per_page,
    )
    return {
        "tool": tool_key,
        "items": [serialize_ai_model(row) for row in rows],
        "pagination": {
            "page": page, "per_page": per_page, "total": total,
            "last_page": max(1, math.ceil(total / per_page)),
        },
    }


@router.post("/general-tools/models/sync/openrouter")
async def sync_openrouter_model_catalog(
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    try:
        return {"success": True, **(await sync_openrouter_models(db))}
    except Exception as exc:
        db.rollback()
        raise _error(exc) from exc


def _text_endpoint(tool: str):
    async def endpoint(
        req: GeneralTextRequest,
        request: Request,
        db: Session = Depends(get_db),
        _: None = Depends(verify_internal_api_key),
    ) -> GeneralToolResponse:
        try:
            return await run_general_text(db, tool, req, request.state.request_id)
        except Exception as exc:
            raise _error(exc) from exc
    return endpoint


router.post("/general-chat", response_model=GeneralToolResponse)(_text_endpoint("general_chat"))
router.post("/general-code", response_model=GeneralToolResponse)(_text_endpoint("general_code"))
router.post("/general-translation", response_model=GeneralToolResponse)(_text_endpoint("general_translation"))


@router.post("/general-media", response_model=GeneralToolResponse)
async def general_media(
    req: GeneralMediaRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> GeneralToolResponse:
    try:
        settings = get_settings()
        get_existing_conversation_for_model(
            db, req.user_id, req.model_id, req.conversation_uuid
        )
        db_model = get_ai_model(
            db, req.selected_model_id, tool_key="general_media", operation="image_generation"
        )
        p = req.state.parameters
        if db_model.provider.lower() != "runware":
            raise ValueError("Selected image model provider is not supported")
        if req.state.model and req.state.model != db_model.provider_model_id:
            raise ValueError("state.model does not match selected_model_id; omit state.model")
        legacy = ImageGeneratorRequest(
            user_id=req.user_id, sub_tool_id=req.model_id,
            conversation_uuid=req.conversation_uuid, user_message=req.user_message,
            provider="runware", model=db_model.provider_model_id, negative_prompt=p.negative_prompt,
            enhance_prompt=p.enhance_prompt, size=p.size, quality=p.quality,
            count=p.count, output_format=p.output_format, seed=p.seed, debug=req.debug,
        )
        result = await run_image_generator(
            db, legacy, request.state.request_id, validate_context=False
        )
        return GeneralToolResponse(
            tool="general_media", provider=result.provider, model=result.model,
            model_tier=db_model.tier, user_id=req.user_id, model_id=req.model_id,
            selected_model_id=req.selected_model_id,
            conversation_uuid=req.conversation_uuid, content=result.message,
            files=result.files, request_id=result.request_id, usage=result.usage,
            cost=result.cost, metadata={"operation": "image_generation", **result.metadata},
            debug=result.debug,
        )
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/general-audio", response_model=GeneralToolResponse)
async def general_audio(
    request: Request,
    payload: str = Form(...),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> GeneralToolResponse:
    try:
        req = GeneralAudioRequest.model_validate_json(payload)
        settings = get_settings()
        get_existing_conversation_for_model(
            db, req.user_id, req.model_id, req.conversation_uuid
        )
        p = req.state.parameters
        operation = req.state.operation
        db_model = get_ai_model(
            db, req.selected_model_id, tool_key="general_audio", operation=operation
        )
        provider = (req.state.provider or (settings.SPEECH_TO_TEXT_PROVIDER if operation == "speech_to_text" else settings.TEXT_TO_SPEECH_PROVIDER)).strip().lower()
        if provider != db_model.provider.lower():
            raise ValueError("Requested provider does not match selected_model_id")
        configured_model = db_model.provider_model_id
        if req.state.model and req.state.model != configured_model:
            raise ValueError("state.model does not match selected_model_id; omit state.model")

        if operation == "speech_to_text":
            if file is None:
                raise ValueError("file is required for speech_to_text")
            result = await run_speech_to_text(
                db, user_id=req.user_id, sub_tool_id=req.model_id,
                conversation_uuid=req.conversation_uuid, file=file,
                provider_override=provider, language=p.language,
                include_segments=p.include_segments, request_id=request.state.request_id,
                model_override=configured_model,
                validate_context=False,
            )
            return GeneralToolResponse(
                tool="general_audio", provider=result.provider, model=result.model,
                model_tier=db_model.tier, user_id=req.user_id, model_id=req.model_id,
                selected_model_id=req.selected_model_id,
                conversation_uuid=req.conversation_uuid, content=result.transcript,
                request_id=result.request_id,
                cost=CostUsage(total_cost=float(result.metadata.get("provider_cost_usd")), currency="USD")
                if result.metadata.get("provider_cost_usd") is not None else None,
                metadata={"operation": operation, "detected_language": result.detected_language, "duration_seconds": result.duration_seconds, "segments": result.segments, **result.metadata},
            )

        tts = TextToSpeechRequest(
            user_id=req.user_id, sub_tool_id=req.model_id,
            conversation_uuid=req.conversation_uuid, user_message=req.user_message,
            provider=provider, model=req.state.model or configured_model,
            voice=p.voice, response_format=p.response_format, speed=p.speed, debug=req.debug,
        )
        result = await run_text_to_speech(
            db, tts, request.state.request_id, validate_context=False
        )
        return GeneralToolResponse(
            tool="general_audio", provider=result.provider, model=result.model,
            model_tier=db_model.tier, user_id=req.user_id, model_id=req.model_id,
            selected_model_id=req.selected_model_id,
            conversation_uuid=req.conversation_uuid, content=result.message,
            files=result.files, request_id=result.request_id, cost=result.cost,
            metadata={"operation": operation, **result.metadata}, debug=result.debug,
        )
    except Exception as exc:
        raise _error(exc) from exc
