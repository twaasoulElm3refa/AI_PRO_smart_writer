from sqlalchemy.orm import Session

from app.settings import get_settings
from app.crud import get_existing_conversation_for_task
from app.memory import load_recent_chat_messages
from app.providers import send_messages_with_model
from app.schemas import ChatMessage, TaskRequest, TaskResponse
from app.tasks import BASE_SYSTEM_PROMPT, TASKS, MODEL_ROUTES
from app.intelligence import (
    resolve_search_enabled,
    temperature_for_task,
    max_tokens_for_task,
)
from app.task_state import (
    update_task_state,
    finalize_task_state,
    build_task_state_context,
)
from app.image_prompt_output import (
    IMAGE_PROMPT_RESPONSE_FORMAT,
    finalize_image_prompt_reply,
    is_valid_image_prompt,
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
    # Image-prompt edits are carried through structured state. Database history is intentionally
    # disabled here so an old malformed model response cannot poison a new prompt request.
    if task_key == "image_prompt_generator":
        return 0

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
    state_context = build_task_state_context(task_key, req.state)

    current_user_message = f"""
    Current user request:
    {clean_user_message}

    {state_context}

    Important:
    Answer the current request above. Use previous conversation only if it is directly relevant.
    """.strip()

    if should_append_current_user_message(history_messages, current_user_message):
        messages.append(ChatMessage(role="user", content=current_user_message))

    return messages


def calculate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    enable_web_search: bool,
    web_search_max_results: int | None,
) -> dict:
    settings = get_settings()

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0

    input_cost = (input_tokens / 1_000_000) * settings.WRITER_INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * settings.WRITER_OUTPUT_COST_PER_1M

    web_search_cost = 0.0
    if enable_web_search:
        search_results = web_search_max_results or settings.WEB_SEARCH_DEFAULT_MAX_RESULTS
        web_search_cost = (search_results / 1000) * settings.WEB_SEARCH_COST_PER_1000_RESULTS

    total_cost = input_cost + output_cost + web_search_cost

    return {
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "web_search_cost": round(web_search_cost, 8),
        "total_cost": round(total_cost, 8),
        "currency": "USD",
    }

async def run_task(db: Session, task_key: str, req: TaskRequest, request_id: str) -> TaskResponse:
    settings = get_settings()

    if task_key not in TASKS:
        raise ValueError(f"Unknown task_key: {task_key}")

    validate_request_limits(req)

    # Normalize and update optional state for legacy one-shot tools.
    # Existing clients can omit state; frontend can now send it like the newer chat tools.
    req.state = update_task_state(task_key, req.state, req.user_message)

    if task_key == "image_prompt_generator" and req.state is not None:
        # Do not send a previous numeric, empty, conversational, or reasoning-leaking value back
        # to the model as last_output.
        previous_output = req.state.get("last_output")
        if previous_output and not is_valid_image_prompt(previous_output):
            req.state["last_output"] = None

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

    if task_key == "image_prompt_generator":
        # This task never needs current web data. Use its dedicated deterministic model settings.
        enable_web_search = False
        route = MODEL_ROUTES[model_key]
        auto_temperature = float(route.get("temperature", settings.DEFAULT_TEMPERATURE))
        auto_max_tokens = int(route.get("max_tokens", settings.DEFAULT_MAX_TOKENS))
    else:
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

    provider_result = await send_messages_with_model(
        model_key=model_key,
        messages=messages,
        temperature_override=temperature_override,
        max_tokens_override=max_tokens_override,
        enable_web_search=enable_web_search,
        web_search_max_results=web_search_max_results,
        web_search_max_total_results=web_search_max_total_results,
        response_format=(
            IMAGE_PROMPT_RESPONSE_FORMAT
            if task_key == "image_prompt_generator"
            else None
        ),
        exclude_reasoning=(task_key == "image_prompt_generator"),
    )
    reply = provider_result.content

    if task_key == "image_prompt_generator":
        reply = finalize_image_prompt_reply(
            reply,
            source_message=req.user_message,
            state=req.state,
        )

    final_state = finalize_task_state(task_key, req.state, reply)

    debug = None
    if req.debug and settings.ENABLE_DEBUG_RESPONSE:
        debug = {
            "state": final_state,
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

    cost = calculate_cost(
        input_tokens=provider_result.input_tokens,
        output_tokens=provider_result.output_tokens,
        enable_web_search=enable_web_search,
        web_search_max_results=web_search_max_results,
    )
    
    return TaskResponse(
        task_key=task_key,
        model_key=model_key,
        user_id=req.user_id,
        sub_tool_id=req.sub_tool_id,
        conversation_uuid=req.conversation_uuid,
        reply=reply,
        request_id=request_id,
        debug=debug,
        usage={
            "input_tokens": provider_result.input_tokens,
            "output_tokens": provider_result.output_tokens,
            "total_tokens": provider_result.total_tokens,
        },
        cost=cost,
        state=final_state,
    )
