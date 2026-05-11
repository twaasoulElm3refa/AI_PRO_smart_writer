from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


RoleType = Literal["system", "user", "assistant"]
SearchMode = Literal["auto", "on", "off"]


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


class TokenUsage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class CostUsage(BaseModel):
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    web_search_cost: Optional[float] = None
    total_cost: Optional[float] = None
    currency: str = "USD"

class TaskOptions(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    history_limit: Optional[int] = None

    # New: smart search control.
    # auto = backend decides, on = force search, off = never search.
    search_mode: SearchMode = "auto"

    # Backward-compatible fields if your frontend already sends them.
    enable_web_search: Optional[bool] = None
    force_no_search: bool = False

    web_search_max_results: Optional[int] = None
    web_search_max_total_results: Optional[int] = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 16:
            raise ValueError("max_tokens must be at least 16")
        if v > 12000:
            raise ValueError("max_tokens is too high")
        return v

    @model_validator(mode="after")
    def normalize_legacy_search_flags(self):
        if self.force_no_search:
            self.search_mode = "off"
        elif self.enable_web_search is True:
            self.search_mode = "on"
        elif self.enable_web_search is False and self.search_mode == "auto":
            # Keep auto when omitted. Only explicit False maps to off if frontend sends it.
            self.search_mode = "off"
        return self


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
    # New
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


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
