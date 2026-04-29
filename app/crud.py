from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User, SubTool, Conversation, Message


def get_user(db: Session, user_id: int) -> User | None:
    stmt = select(User).where(User.id == user_id, User.deleted_at.is_(None))
    return db.execute(stmt).scalar_one_or_none()


def get_sub_tool(db: Session, sub_tool_id: int) -> SubTool | None:
    stmt = select(SubTool).where(SubTool.id == sub_tool_id, SubTool.deleted_at.is_(None))
    return db.execute(stmt).scalar_one_or_none()


def get_conversation_by_uuid_for_user(db: Session, conversation_uuid: str, user_id: int) -> Conversation | None:
    stmt = select(Conversation).where(
        Conversation.uuid == conversation_uuid,
        Conversation.user_id == user_id,
        Conversation.deleted_at.is_(None),
    )
    return db.execute(stmt).scalar_one_or_none()


def create_conversation(
    db: Session,
    user_id: int,
    sub_tool_id: int,
    conversation_uuid: str,
    is_pinned: bool = False,
    is_archived: bool = False,
) -> Conversation:
    now = datetime.utcnow()
    conv = Conversation(
        user_id=user_id,
        sub_tool_id=sub_tool_id,
        uuid=conversation_uuid,
        is_pinned=is_pinned,
        is_archived=is_archived,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )
    db.add(conv)
    db.flush()
    return conv


def ensure_conversation(
    db: Session,
    user_id: int,
    sub_tool_id: int,
    conversation_uuid: str,
) -> Conversation:
    conv = get_conversation_by_uuid_for_user(db, conversation_uuid, user_id)
    if conv:
        if conv.sub_tool_id != sub_tool_id:
            raise ValueError("Conversation sub_tool_id does not match request")
        return conv

    user = get_user(db, user_id)
    if not user:
        raise ValueError("User not found")

    sub_tool = get_sub_tool(db, sub_tool_id)
    if not sub_tool:
        raise ValueError("Sub tool not found")

    return create_conversation(
        db=db,
        user_id=user_id,
        sub_tool_id=sub_tool_id,
        conversation_uuid=conversation_uuid,
    )


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


def save_message(
    db: Session,
    conversation_id: int,
    role: str,
    content: str,
    is_error: bool = False,
) -> Message:
    now = datetime.utcnow()
    msg = Message(
        conversation_id=conversation_id,
        content=content,
        is_error=is_error,
        role=role,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )
    db.add(msg)
    db.flush()
    return msg


def touch_conversation(db: Session, conv: Conversation) -> None:
    conv.updated_at = datetime.utcnow()
    db.add(conv)
    db.flush()