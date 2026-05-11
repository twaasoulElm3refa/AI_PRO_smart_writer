from typing import List
from sqlalchemy.orm import Session

from app.crud import get_recent_messages
from app.schemas import ChatMessage


def normalize_db_role(role: str | None) -> str | None:
    """
    Laravel projects sometimes save assistant replies as 'assistant' or 'ai'.
    OpenRouter/OpenAI chat format requires: system, user, assistant.
    """
    if not role:
        return None

    normalized = role.strip().lower()

    if normalized in ("system", "user", "assistant"):
        return normalized

    if normalized == "ai":
        return "assistant"

    return None


def load_recent_chat_messages(
    db: Session,
    conversation_id: int,
    limit: int,
) -> List[ChatMessage]:
    rows = get_recent_messages(db, conversation_id, limit)

    messages: List[ChatMessage] = []
    for row in rows:
        role = normalize_db_role(row.role)
        content = (row.content or "").strip()

        if role and content:
            messages.append(ChatMessage(role=role, content=content))

    return messages
