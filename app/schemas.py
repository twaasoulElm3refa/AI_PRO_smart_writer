from typing import Any, Dict, List, Literal, Optional
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

    search_mode: SearchMode = "auto"
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
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


# These are examples for the UI/prompt, not strict backend enums.
# Keep them as normal strings so the chat can accept any language, tone, goal, etc.
HeadlineContentType = str
HeadlineGoal = str
HeadlineLanguage = str
HeadlineTone = str
HeadlineLength = str

HeadlineChatType = Literal[
    "question",
    "result",
]


class HeadlineState(BaseModel):
    content: Optional[str] = None
    content_type: Optional[HeadlineContentType] = None
    goal: Optional[HeadlineGoal] = None
    language: Optional[HeadlineLanguage] = None
    tone: Optional[HeadlineTone] = None
    number_of_headlines: Optional[int] = None
    headline_length: Optional[HeadlineLength] = None
    extra_options: List[str] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = str(v).strip()
        return v or None

    @field_validator("content_type", "goal", "language", "tone", "headline_length", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("number_of_headlines", mode="before")
    @classmethod
    def validate_number_of_headlines(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None

        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("number_of_headlines must be a positive integer")

        if value < 1:
            raise ValueError("number_of_headlines must be at least 1")

        # Not restricted to fixed choices. This cap only protects cost/performance.
        # Increase it if you want to allow larger batches.
        if value > 100:
            raise ValueError("number_of_headlines is too high; maximum is 100")

        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class HeadlineChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[HeadlineState] = None
    debug: bool = False
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


class HeadlineItem(BaseModel):
    id: int
    text: str
    subheadline: Optional[str] = None


class HeadlineChatResponse(BaseModel):
    success: bool = True
    type: HeadlineChatType
    tool: str = "ai_headline_generator"
    provider: str = "openrouter"
    model_key: str = "headline_fast"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: HeadlineState
    headlines: List[HeadlineItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


# These are examples for the UI/prompt, not strict backend enums.
# Keep them as normal strings so the chat can accept any language, tone, mode, etc.
ParaphraserLanguage = str
ParaphraserTone = str
ParaphraserRewriteMode = str
ParaphraserChangeLevel = str

ParaphraserChatType = Literal[
    "question",
    "result",
]


class ParaphraserState(BaseModel):
    content: Optional[str] = None
    language: Optional[ParaphraserLanguage] = "Auto Detect"
    tone: Optional[ParaphraserTone] = "Professional"
    rewrite_mode: Optional[ParaphraserRewriteMode] = "Paraphrase"
    change_level: Optional[ParaphraserChangeLevel] = "Medium"
    results_count: Optional[int] = 1
    extra_options: List[str] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = str(v).strip()
        return v or None

    @field_validator("language", "tone", "rewrite_mode", "change_level", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None

        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")

        if value < 1:
            raise ValueError("results_count must be at least 1")

        # Not restricted to fixed choices. This cap only protects cost/performance.
        if value > 20:
            raise ValueError("results_count is too high; maximum is 20")

        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class ParaphraserChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[ParaphraserState] = None
    debug: bool = False
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


class ParaphraserResultItem(BaseModel):
    id: int
    text: str


class ParaphraserChatResponse(BaseModel):
    success: bool = True
    type: ParaphraserChatType
    tool: str = "ai_paraphraser"
    provider: str = "openrouter"
    model_key: str = "paraphraser_fast"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: ParaphraserState
    results: List[ParaphraserResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
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
