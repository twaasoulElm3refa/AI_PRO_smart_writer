from typing import List
from sqlalchemy.orm import Session

from app.crud import get_recent_messages
from app.schemas import ChatMessage


def load_recent_chat_messages(
    db: Session,
    conversation_id: int,
    limit: int,
) -> List[ChatMessage]:
    rows = get_recent_messages(db, conversation_id, limit)

    messages: List[ChatMessage] = []
    for row in rows:
        if row.role in ("user", "assistant", "system"):
            messages.append(ChatMessage(role=row.role, content=row.content))
    return messages