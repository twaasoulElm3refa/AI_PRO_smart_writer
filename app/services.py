from sqlalchemy.orm import Session

from app.settings import get_settings
from app.crud import get_existing_conversation_for_task
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
        raise ValueError(
            f"user_message exceeds max length of {settings.MAX_USER_MESSAGE_LENGTH}"
        )

    if req.task_options and req.task_options.history_limit is not None:
        if req.task_options.history_limit < 0:
            raise ValueError("history_limit cannot be negative")
        if req.task_options.history_limit > settings.MAX_HISTORY_MESSAGES:
            raise ValueError(
                f"history_limit exceeds max value of {settings.MAX_HISTORY_MESSAGES}"
            )


def resolve_history_limit(task_key: str, req: TaskRequest) -> int:
    task = TASKS[task_key]
    settings = get_settings()

    default_history_limit = int(
        task.get("history_limit", settings.DEFAULT_HISTORY_LIMIT)
    )

    if req.task_options and req.task_options.history_limit is not None:
        return int(req.task_options.history_limit)

    return min(default_history_limit, settings.MAX_HISTORY_MESSAGES)


def should_append_current_user_message(
    history_messages: list[ChatMessage],
    current_user_message: str,
) -> bool:
    """
    If Laravel saves the user message before calling FastAPI, the latest DB message
    may already be exactly the same user message. In that case, do not append it
    again to the provider payload, otherwise the AI receives duplicated input.
    """
    if not history_messages:
        return True

    last = history_messages[-1]
    return not (
        last.role == "user"
        and last.content.strip() == current_user_message.strip()
    )


def build_messages_for_task(
    db: Session,
    task_key: str,
    req: TaskRequest,
) -> list[ChatMessage]:
    if task_key not in TASKS:
        raise ValueError(f"Unknown task_key: {task_key}")

    conversation = get_existing_conversation_for_task(
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

    history_limit = resolve_history_limit(task_key, req)
    history_messages = load_recent_chat_messages(
        db=db,
        conversation_id=conversation.id,
        limit=history_limit,
    )

    messages.extend(history_messages)

    clean_user_message = req.user_message.strip()
    if should_append_current_user_message(history_messages, clean_user_message):
        messages.append(ChatMessage(role="user", content=clean_user_message))

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

    # Read/validate only. No insert, no update, no commit.
    get_existing_conversation_for_task(
        db=db,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
    )

    temperature_override = req.task_options.temperature if req.task_options else None
    max_tokens_override = req.task_options.max_tokens if req.task_options else None

    messages = build_messages_for_task(db, task_key, req)

    reply = await send_messages_with_model(
        model_key=model_key,
        messages=messages,
        temperature_override=temperature_override,
        max_tokens_override=max_tokens_override,
    )

    debug = None
    if req.debug and settings.ENABLE_DEBUG_RESPONSE:
        debug = {
            "final_system_prompt": messages[0].content,
            "messages_sent": [m.model_dump() for m in messages],
            "client_metadata": req.client_metadata,
            "db_write_mode": "disabled_fastapi_read_only",
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
