from sqlalchemy.orm import Session

from app.settings import get_settings
from app.crud import ensure_conversation, save_message, touch_conversation
from app.memory import load_recent_chat_messages
from app.providers import send_messages_with_model
from app.schemas import ChatMessage, TaskRequest, TaskResponse
from app.tasks import BASE_SYSTEM_PROMPT, TASKS


def build_system_prompt(
    task_key: str,
    custom_prompt: str | None = None,
    response_language: str | None = None,
) -> str:
    if task_key not in TASKS:
        raise ValueError(f"Unknown task_key: {task_key}")

    task = TASKS[task_key]

    parts = [
        BASE_SYSTEM_PROMPT,
        "",
        f"Current task: {task_key}",
        f"Task description: {task['description']}",
        "",
        "Task-specific instructions:",
        task["system_prompt"],
    ]

    if response_language:
        parts.extend([
            "",
            f"Preferred response language: {response_language}",
        ])

    if custom_prompt:
        parts.extend([
            "",
            "Additional instructions:",
            custom_prompt.strip(),
        ])

    return "\n".join(parts).strip()


def validate_request_limits(req: TaskRequest) -> None:
    settings = get_settings()

    if len(req.user_message) > settings.MAX_USER_MESSAGE_LENGTH:
        raise ValueError(f"user_message exceeds max length of {settings.MAX_USER_MESSAGE_LENGTH}")


def build_messages_for_task(
    db: Session,
    task_key: str,
    req: TaskRequest,
) -> list[ChatMessage]:
    if task_key not in TASKS:
        raise ValueError(f"Unknown task_key: {task_key}")

    task = TASKS[task_key]
    default_history_limit = task.get("history_limit", 12)

    history_limit = default_history_limit
    if req.task_options and req.task_options.history_limit is not None:
        history_limit = req.task_options.history_limit

    conversation = ensure_conversation(
        db=db,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
    )

    final_system_prompt = build_system_prompt(
        task_key=task_key,
        custom_prompt=req.custom_prompt,
        response_language=req.response_language,
    )

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=final_system_prompt)
    ]

    messages.extend(load_recent_chat_messages(db, conversation.id, history_limit))
    messages.append(ChatMessage(role="user", content=req.user_message.strip()))

    return messages


async def run_task(
    db: Session,
    task_key: str,
    req: TaskRequest,
    request_id: str,
) -> TaskResponse:
    settings = get_settings()

    if task_key not in TASKS:
        raise ValueError(f"Unknown task_key: {task_key}")

    validate_request_limits(req)

    task = TASKS[task_key]
    model_key = task["model_key"]

    conversation = ensure_conversation(
        db=db,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
    )

    temperature_override = req.task_options.temperature if req.task_options else None
    max_tokens_override = req.task_options.max_tokens if req.task_options else None

    messages = build_messages_for_task(db, task_key, req)

    save_message(
        db=db,
        conversation_id=conversation.id,
        role="user",
        content=req.user_message.strip(),
        is_error=False,
    )

    reply = await send_messages_with_model(
        model_key=model_key,
        messages=messages,
        temperature_override=temperature_override,
        max_tokens_override=max_tokens_override,
    )

    save_message(
        db=db,
        conversation_id=conversation.id,
        role="assistant",
        content=reply,
        is_error=False,
    )

    touch_conversation(db, conversation)
    db.commit()

    debug = None
    if req.debug and settings.ENABLE_DEBUG_RESPONSE:
        debug = {
            "final_system_prompt": messages[0].content,
            "messages_sent": [m.model_dump() for m in messages],
            "client_metadata": req.client_metadata,
        }

    return TaskResponse(
        reply=reply,
        task_key=task_key,
        model_key=model_key,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
        request_id=request_id,
        debug=debug,
    )