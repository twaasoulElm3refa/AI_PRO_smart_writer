import uuid
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.settings import get_settings
from app.crud import create_conversation, get_conversation_by_uuid_for_user, list_conversations_for_user, list_messages
from app.database import get_db
from app.schemas import (
    TaskRequest,
    TaskResponse,
    ConversationCreateRequest,
    ConversationResponse,
    MessageResponse,
)
from app.services import run_task
from app.tasks import TASKS
from app.security import verify_internal_api_key

settings = get_settings()

app = FastAPI(title=settings.APP_NAME)

origins = ["*"] if settings.ALLOWED_ORIGINS == "*" else [
    origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()
]

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
    }


@app.post("/conversations", response_model=ConversationResponse, tags=["conversations"])
async def create_conversation_endpoint(
    req: ConversationCreateRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    try:
        existing = get_conversation_by_uuid_for_user(db, req.conversation_uuid, req.user_id)
        if existing:
            return ConversationResponse(
                id=existing.id,
                user_id=existing.user_id,
                sub_tool_id=existing.sub_tool_id,
                uuid=existing.uuid,
                is_pinned=bool(existing.is_pinned),
                is_archived=bool(existing.is_archived),
                created_at=existing.created_at.isoformat() if existing.created_at else None,
                updated_at=existing.updated_at.isoformat() if existing.updated_at else None,
            )

        conv = create_conversation(
            db=db,
            user_id=req.user_id,
            sub_tool_id=req.sub_tool_id,
            conversation_uuid=req.conversation_uuid,
            is_pinned=req.is_pinned,
            is_archived=req.is_archived,
        )
        db.commit()
        db.refresh(conv)

        return ConversationResponse(
            id=conv.id,
            user_id=conv.user_id,
            sub_tool_id=conv.sub_tool_id,
            uuid=conv.uuid,
            is_pinned=bool(conv.is_pinned),
            is_archived=bool(conv.is_archived),
            created_at=conv.created_at.isoformat() if conv.created_at else None,
            updated_at=conv.updated_at.isoformat() if conv.updated_at else None,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=repr(e))



@app.get("/users/{user_id}/conversations/{conversation_uuid}/messages", response_model=list[MessageResponse], tags=["messages"])
async def list_conversation_messages_endpoint(user_id: int, conversation_uuid: str, db: Session = Depends(get_db)):
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
            db.rollback()
            raise HTTPException(status_code=400, detail=str(e))
        except httpx.HTTPStatusError as e:
            db.rollback()
            raise HTTPException(status_code=502, detail=f"Provider HTTP error: {e.response.text}")
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=502, detail=str(e))

    return endpoint


for task_key, task in TASKS.items():
    app.post(task["path"], response_model=TaskResponse, tags=["tasks"])(
        create_task_endpoint(task_key)
    )
