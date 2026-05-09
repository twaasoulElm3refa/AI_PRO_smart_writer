from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Tools API"
    APP_ENV: str = "production"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 9000
    ALLOWED_ORIGINS: str = "*"

    # Use your real production DB here, for example:
    # mysql+pymysql://DB_USER:DB_PASSWORD@127.0.0.1:3306/DB_NAME?charset=utf8mb4
    DATABASE_URL: str = Field(default="")

    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Used for OpenRouter headers only.
    SITE_URL: str = "https://pro.aiarabic.com"

    REQUEST_TIMEOUT_SECONDS: float = 120.0

    DEFAULT_TEMPERATURE: float = 0.45
    DEFAULT_MAX_TOKENS: int = 2500
    DEFAULT_HISTORY_LIMIT: int = 8

    ENABLE_DEBUG_RESPONSE: bool = False
    MAX_USER_MESSAGE_LENGTH: int = 20000
    MAX_HISTORY_MESSAGES: int = 12

    # Writer should be strong. You can use any OpenRouter model here.
    OPENROUTER_WRITER_MODEL: str = "deepseek/deepseek-v3.2"
    OPENROUTER_SUMMARIZER_MODEL: str = "deepseek/deepseek-v3.2"
    OPENROUTER_HEADLINE_MODEL: str = "deepseek/deepseek-v3.2"
    OPENROUTER_PARAPHRASER_MODEL: str = "deepseek/deepseek-v3.2"

    # Web search defaults. Keep small for cost control.
    WEB_SEARCH_DEFAULT_MAX_RESULTS: int = 3
    WEB_SEARCH_DEFAULT_MAX_TOTAL_RESULTS: int = 5
    WEB_SEARCH_HARD_MAX_RESULTS: int = 5
    WEB_SEARCH_HARD_MAX_TOTAL_RESULTS: int = 10

    INTERNAL_API_KEY: str = Field(default="")
    INTERNAL_API_HEADER_NAME: str = "X-Internal-Api-Key"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
