from datetime import datetime
from sqlalchemy import BigInteger, String, Text, DateTime, Boolean, ForeignKey, JSON, Numeric
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.settings import get_settings


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SubTool(Base):
    __tablename__ = "sub_tools"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    main_tool_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    website: Mapped[str | None] = mapped_column(String(50), nullable=True)
    image: Mapped[str | None] = mapped_column(String(255), nullable=True)
    slug: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_placeholder: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    sort_order: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    sub_tool_id: Mapped[int] = mapped_column(ForeignKey("sub_tools.id"), index=True)
    uuid: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    is_pinned: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool | None] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.id",
    )


class ModelsConversation(Base):
    """Conversation table used only by the new general-model tools.

    The table name intentionally preserves the spelling used by the Laravel/MySQL
    backend: ``models_converstaions``.
    """

    __tablename__ = get_settings().MODELS_CONVERSATIONS_TABLE

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    model_id: Mapped[int] = mapped_column(BigInteger, index=True)
    uuid: Mapped[str] = mapped_column(String(255), unique=True, index=True)


class AIModel(Base):
    """Unified provider-model catalog for text, image, audio and video tools."""
    __tablename__ = "ai_models"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(50), index=True)
    provider_model_id: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_key: Mapped[str] = mapped_column(String(100), index=True)
    operation: Mapped[str] = mapped_column(String(100), index=True)
    tier: Mapped[str] = mapped_column(String(30), default="standard", index=True)
    capabilities: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    parameter_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pricing: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    provider_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    sort_order: Mapped[int] = mapped_column(default=0)
    pricing_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    provider_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    is_error: Mapped[bool | None] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String(20), index=True)  # user / system / assistant
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
