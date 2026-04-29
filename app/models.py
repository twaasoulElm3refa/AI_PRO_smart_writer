from datetime import datetime
from sqlalchemy import BigInteger, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


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