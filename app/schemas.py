from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


RoleType = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: RoleType
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty")
        return v


class TaskOptions(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    history_limit: Optional[int] = None


class TaskRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    custom_prompt: Optional[str] = None
    response_language: Optional[str] = None
    debug: bool = False
    task_options: Optional[TaskOptions] = None
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class TaskResponse(BaseModel):
    reply: str
    task_key: str
    model_key: str
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    request_id: str
    debug: Optional[dict] = None


class ConversationCreateRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    is_pinned: bool = False
    is_archived: bool = False


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    sub_tool_id: int
    uuid: str
    is_pinned: bool
    is_archived: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    is_error: bool
    created_at: Optional[str] = None