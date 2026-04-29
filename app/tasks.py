from typing import Dict


BASE_SYSTEM_PROMPT = """
You are a helpful AI assistant.

Rules:
- Answer the user's request clearly and directly.
- Match the user's language unless instructed otherwise.
- Stay focused on the requested task.
- Do not invent facts.
- Do not add meta commentary unless requested.
- If the request is too unclear to answer well, ask only the minimum necessary clarification.
- If the request is actionable and reasonably clear, answer directly.
""".strip()


MODEL_ROUTES: Dict[str, dict] = {
    "writer_fast": {
        "provider": "openrouter",
        "model_env_key": "OPENROUTER_WRITER_MODEL",
        "temperature": 0.4,
        "max_tokens": 1000,
    },
    "summarizer_fast": {
        "provider": "openrouter",
        "model_env_key": "OPENROUTER_SUMMARIZER_MODEL",
        "temperature": 0.3,
        "max_tokens": 700,
    },
    "headline_fast": {
        "provider": "openrouter",
        "model_env_key": "OPENROUTER_HEADLINE_MODEL",
        "temperature": 0.6,
        "max_tokens": 400,
    },
    "paraphraser_fast": {
        "provider": "openrouter",
        "model_env_key": "OPENROUTER_PARAPHRASER_MODEL",
        "temperature": 0.35,
        "max_tokens": 900,
    },
}


TASKS: Dict[str, dict] = {
    "writer": {
        "path": "/tasks/writer",
        "description": "Write polished, clear, ready-to-use content.",
        "system_prompt": """
You help the user write polished, clear, readable, and useful content.
Improve structure, tone, and clarity.
Deliver final content directly unless clarification is truly necessary.
""".strip(),
        "model_key": "writer_fast",
        "history_limit": 12,
    },
    "summarizer": {
        "path": "/tasks/summarizer",
        "description": "Summarize the user's text clearly and accurately.",
        "system_prompt": """
You summarize the provided content faithfully and clearly.
Do not add new facts.
Keep the summary concise unless the user asks for more detail.
""".strip(),
        "model_key": "summarizer_fast",
        "history_limit": 10,
    },
    "headline_generator": {
        "path": "/tasks/headline-generator",
        "description": "Generate strong, clear, relevant headlines.",
        "system_prompt": """
Generate strong, clear, relevant headlines based on the user's content.
Avoid misleading clickbait.
If the user does not specify a number, provide 5 options.
""".strip(),
        "model_key": "headline_fast",
        "history_limit": 8,
    },
    "paraphraser": {
        "path": "/tasks/paraphraser",
        "description": "Rewrite text while preserving the original meaning.",
        "system_prompt": """
Rewrite the user's text while preserving the original meaning.
Improve clarity and fluency.
Do not add new facts or ideas unless the user asks.
""".strip(),
        "model_key": "paraphraser_fast",
        "history_limit": 10,
    },
}