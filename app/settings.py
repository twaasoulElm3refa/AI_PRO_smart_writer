from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Tools API"
    APP_ENV: str = "production"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    ALLOWED_ORIGINS: str = "*"

    DATABASE_URL: str = Field(default="")

    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    REQUEST_TIMEOUT_SECONDS: float = 60.0

    DEFAULT_TEMPERATURE: float = 0.4
    DEFAULT_MAX_TOKENS: int = 800
    DEFAULT_HISTORY_LIMIT: int = 12

    ENABLE_DEBUG_RESPONSE: bool = False
    MAX_USER_MESSAGE_LENGTH: int = 12000
    MAX_HISTORY_MESSAGES: int = 20

    OPENROUTER_WRITER_MODEL: str = "deepseek/deepseek-v3.2"
    OPENROUTER_SUMMARIZER_MODEL: str = "deepseek/deepseek-v3.2"
    OPENROUTER_HEADLINE_MODEL: str = "deepseek/deepseek-v3.2"
    OPENROUTER_PARAPHRASER_MODEL: str = "deepseek/deepseek-v3.2"

    INTERNAL_API_KEY: str = Field(default="")
    INTERNAL_API_HEADER_NAME: str = "test123"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()