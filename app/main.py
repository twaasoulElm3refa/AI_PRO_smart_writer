import uuid
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.settings import get_settings
from app.crud import get_conversation_by_uuid_for_user, list_messages
from app.database import get_db
from app.schemas import (
    TaskRequest,
    TaskResponse,
    MessageResponse,
    HeadlineChatRequest,
    HeadlineChatResponse,
    ParaphraserChatRequest,
    ParaphraserChatResponse,
)
from app.headline_chat import run_headline_chat
from app.paraphraser_chat import run_paraphraser_chat
from app.services import run_task
from app.tasks import TASKS
from app.security import verify_internal_api_key

settings = get_settings()

app = FastAPI(title=settings.APP_NAME)

origins = (
    ["*"]
    if settings.ALLOWED_ORIGINS == "*"
    else [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "db_write_mode": "disabled_fastapi_read_only",
    }


@app.get(
    "/users/{user_id}/conversations/{conversation_uuid}/messages",
    response_model=list[MessageResponse],
    tags=["messages"],
)
async def list_conversation_messages_endpoint(
    user_id: int,
    conversation_uuid: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    conv = get_conversation_by_uuid_for_user(db, conversation_uuid, user_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    rows = list_messages(db, conv.id, limit=500)
    return [
        MessageResponse(
            id=row.id,
            conversation_id=row.conversation_id,
            role=row.role,
            content=row.content,
            is_error=bool(row.is_error),
            created_at=row.created_at.isoformat() if row.created_at else None,
        )
        for row in rows
    ]


def create_task_endpoint(task_key: str):
    async def endpoint(
        req: TaskRequest,
        request: Request,
        db: Session = Depends(get_db),
        _: None = Depends(verify_internal_api_key),
    ) -> TaskResponse:
        try:
            return await run_task(db, task_key, req, request.state.request_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except httpx.HTTPStatusError as e:
            provider_text = e.response.text if e.response is not None else str(e)
            raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    return endpoint


@app.post(
    "/tasks/headline-generator/chat",
    response_model=HeadlineChatResponse,
    tags=["tasks"],
)
async def headline_generator_chat_endpoint(
    req: HeadlineChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> HeadlineChatResponse:
    try:
        return await run_headline_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post(
    "/tasks/paraphraser/chat",
    response_model=ParaphraserChatResponse,
    tags=["tasks"],
)
async def paraphraser_chat_endpoint(
    req: ParaphraserChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> ParaphraserChatResponse:
    try:
        return await run_paraphraser_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

for task_key, task in TASKS.items():
    app.post(task["path"], response_model=TaskResponse, tags=["tasks"])(
        create_task_endpoint(task_key)
    )
