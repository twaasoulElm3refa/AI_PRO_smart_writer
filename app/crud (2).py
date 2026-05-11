from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User, SubTool, Conversation, Message


def get_user(db: Session, user_id: int) -> User | None:
    stmt = select(User).where(
        User.id == user_id,
        User.deleted_at.is_(None),
    )
    return db.execute(stmt).scalar_one_or_none()


def get_sub_tool(db: Session, sub_tool_id: int) -> SubTool | None:
    stmt = select(SubTool).where(
        SubTool.id == sub_tool_id,
        SubTool.deleted_at.is_(None),
    )
    return db.execute(stmt).scalar_one_or_none()


def get_conversation_by_uuid_for_user(
    db: Session,
    conversation_uuid: str,
    user_id: int,
) -> Conversation | None:
    stmt = select(Conversation).where(
        Conversation.uuid == conversation_uuid,
        Conversation.user_id == user_id,
        Conversation.deleted_at.is_(None),
    )
    return db.execute(stmt).scalar_one_or_none()


def get_existing_conversation_for_task(
    db: Session,
    user_id: int,
    sub_tool_id: int,
    conversation_uuid: str,
) -> Conversation:
    """
    Read-only conversation resolver.

    FastAPI must NOT create conversations and must NOT insert/update messages.
    Laravel should create/find the conversation and save messages.
    This function only validates that the requested conversation exists and belongs
    to the same user and sub-tool.
    """
    user = get_user(db, user_id)
    if not user:
        raise ValueError("User not found")

    sub_tool = get_sub_tool(db, sub_tool_id)
    if not sub_tool:
        raise ValueError("Sub tool not found")

    conversation = get_conversation_by_uuid_for_user(
        db=db,
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    if not conversation:
        raise ValueError(
            "Conversation not found. Create it in Laravel before calling FastAPI."
        )

    if int(conversation.sub_tool_id) != int(sub_tool_id):
        raise ValueError("Conversation sub_tool_id does not match request")

    return conversation


def list_conversations_for_user(db: Session, user_id: int) -> list[Conversation]:
    stmt = (
        select(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
    )
    return list(db.execute(stmt).scalars().all())


def get_recent_messages(db: Session, conversation_id: int, limit: int) -> list[Message]:
    limit = max(0, int(limit or 0))
    if limit <= 0:
        return []

    stmt = (
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )
        .order_by(Message.id.desc())
        .limit(limit)
    )
    rows = list(db.execute(stmt).scalars().all())
    rows.reverse()
    return rows


def list_messages(db: Session, conversation_id: int, limit: int = 500) -> list[Message]:
    stmt = (
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )
        .order_by(Message.id.asc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())
