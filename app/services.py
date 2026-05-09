from sqlalchemy.orm import Session

from app.settings import get_settings
from app.crud import get_existing_conversation_for_task
from app.memory import load_recent_chat_messages
from app.providers import send_messages_with_model
from app.schemas import ChatMessage, TaskRequest, TaskResponse
from app.tasks import BASE_SYSTEM_PROMPT, TASKS
from app.intelligence import (
    resolve_search_enabled,
    temperature_for_task,
    max_tokens_for_task,
)


def build_system_prompt(
    task_key: str,
    custom_prompt: str | None = None,
    response_language: str | None = None,
    enable_web_search: bool = False,
    search_reason: str | None = None,
) -> str:
    if task_key not in TASKS:
        raise ValueError(f"Unknown task_key: {task_key}")

    task = TASKS[task_key]
    task_prompt = task.get("search_system_prompt") if enable_web_search else task.get("system_prompt")
    task_prompt = task_prompt or task.get("system_prompt", "")

    parts = [
        BASE_SYSTEM_PROMPT,
        "",
        f"Current task: {task_key}",
        f"Task description: {task['description']}",
        "",
        "Task-specific instructions:",
        task_prompt,
    ]

    if enable_web_search:
        parts.extend([
            "",
            "Fresh-data mode: ON",
            f"Search reason: {search_reason or 'not specified'}",
            "Use search/context for factual/current claims. Do not guess.",
        ])
    else:
        parts.extend([
            "",
            "Fresh-data mode: OFF",
            "Use only the user-provided content and general writing skill. Do not add unsupported current facts.",
        ])

    if response_language:
        parts.extend(["", f"Preferred response language: {response_language}"])

    if custom_prompt:
        parts.extend(["", "Additional instructions:", custom_prompt.strip()])

    return "\n".join(parts).strip()


def validate_request_limits(req: TaskRequest) -> None:
    settings = get_settings()

    if len(req.user_message) > settings.MAX_USER_MESSAGE_LENGTH:
        raise ValueError(f"user_message exceeds max length of {settings.MAX_USER_MESSAGE_LENGTH}")

    if req.task_options and req.task_options.history_limit is not None:
        if req.task_options.history_limit < 0:
            raise ValueError("history_limit cannot be negative")
        if req.task_options.history_limit > settings.MAX_HISTORY_MESSAGES:
            raise ValueError(f"history_limit exceeds max value of {settings.MAX_HISTORY_MESSAGES}")


def resolve_history_limit(task_key: str, req: TaskRequest) -> int:
    task = TASKS[task_key]
    settings = get_settings()
    default_history_limit = int(task.get("history_limit", settings.DEFAULT_HISTORY_LIMIT))

    if req.task_options and req.task_options.history_limit is not None:
        return int(req.task_options.history_limit)

    return min(default_history_limit, settings.MAX_HISTORY_MESSAGES)


def should_append_current_user_message(history_messages: list[ChatMessage], current_user_message: str) -> bool:
    if not history_messages:
        return True
    last = history_messages[-1]
    return not (last.role == "user" and last.content.strip() == current_user_message.strip())


def build_messages_for_task(
    db: Session,
    task_key: str,
    req: TaskRequest,
    enable_web_search: bool,
    search_reason: str | None,
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
        enable_web_search=enable_web_search,
        search_reason=search_reason,
    )

    messages: list[ChatMessage] = [ChatMessage(role="system", content=final_system_prompt)]

    history_limit = resolve_history_limit(task_key, req)
    history_messages = load_recent_chat_messages(db=db, conversation_id=conversation.id, limit=history_limit)
    messages.extend(history_messages)

    clean_user_message = req.user_message.strip()
    if should_append_current_user_message(history_messages, clean_user_message):
        messages.append(ChatMessage(role="user", content=clean_user_message))

    return messages


async def run_task(db: Session, task_key: str, req: TaskRequest, request_id: str) -> TaskResponse:
    settings = get_settings()

    if task_key not in TASKS:
        raise ValueError(f"Unknown task_key: {task_key}")

    validate_request_limits(req)

    task = TASKS[task_key]
    model_key = task["model_key"]

    # Read/validate only. Laravel should create conversation and save messages.
    get_existing_conversation_for_task(
        db=db,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
    )

    options = req.task_options
    search_mode = options.search_mode if options else "auto"
    enable_web_search, analysis = resolve_search_enabled(req.user_message, search_mode)

    auto_temperature = temperature_for_task(analysis.task_kind, enable_web_search)
    auto_max_tokens = max_tokens_for_task(analysis.task_kind)

    temperature_override = options.temperature if options and options.temperature is not None else auto_temperature
    max_tokens_override = options.max_tokens if options and options.max_tokens is not None else auto_max_tokens

    web_search_max_results = options.web_search_max_results if options else None
    web_search_max_total_results = options.web_search_max_total_results if options else None

    messages = build_messages_for_task(
        db=db,
        task_key=task_key,
        req=req,
        enable_web_search=enable_web_search,
        search_reason=analysis.reason,
    )

    reply = await send_messages_with_model(
        model_key=model_key,
        messages=messages,
        temperature_override=temperature_override,
        max_tokens_override=max_tokens_override,
        enable_web_search=enable_web_search,
        web_search_max_results=web_search_max_results,
        web_search_max_total_results=web_search_max_total_results,
    )

    debug = None
    if req.debug and settings.ENABLE_DEBUG_RESPONSE:
        debug = {
            "analysis": {
                "task_kind": analysis.task_kind,
                "needs_search": analysis.needs_search,
                "reason": analysis.reason,
                "final_search_enabled": enable_web_search,
            },
            "model": {
                "model_key": model_key,
                "temperature": temperature_override,
                "max_tokens": max_tokens_override,
            },
            "web_search": {
                "enabled": enable_web_search,
                "max_results": web_search_max_results or settings.WEB_SEARCH_DEFAULT_MAX_RESULTS,
                "max_total_results": web_search_max_total_results or settings.WEB_SEARCH_DEFAULT_MAX_TOTAL_RESULTS,
            },
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
