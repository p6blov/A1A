"""Centralized settings, loaded from environment variables / .env file.

Keeping all config here means no module reads os.environ directly — that makes
testing easier and keeps secrets out of the rest of the codebase.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Primary provider: OpenRouter ---------------------------------------
    # One key fronts several labs' models, which is what makes a free-tier
    # fan-out a genuine cross-vendor second opinion. It speaks the OpenAI
    # chat-completions protocol, so OpenAICompatProvider covers it unchanged.
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="openai/gpt-oss-20b:free", alias="OPENROUTER_MODEL"
    )
    # OpenRouter's optional attribution headers, shown on their activity page.
    # Neither identifies the end user — they describe this app, not its callers.
    openrouter_site_url: str = Field(
        default="http://localhost:5173", alias="OPENROUTER_SITE_URL"
    )
    openrouter_app_title: str = Field(default="A1A", alias="OPENROUTER_APP_TITLE")

    # Upper bound on a client-supplied agent roster (see /api/chat). Each agent
    # is one upstream call per turn, so this caps what a single request can
    # spend — and how fast it burns the free tier's daily quota.
    max_custom_agents: int = Field(default=6, alias="MAX_CUSTOM_AGENTS")

    # --- Direct first-party providers (dormant until you opt in) ------------
    # Each talks straight to its vendor's own API with its own key.
    # NOTE: these are not in any task line-up in tasks.yaml; adding a key alone
    # does nothing until you also list the provider on a task.
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-3.1-flash-lite", alias="GOOGLE_MODEL")

    # A second Gemini model under the same Google key, for a free-tier fan-out
    # that does not go through OpenRouter.
    google_secondary_model: str = Field(
        default="gemini-3-flash-preview", alias="GOOGLE_SECONDARY_MODEL"
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5-mini", alias="OPENAI_MODEL")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-opus-4-8", alias="ANTHROPIC_MODEL")

    # Defaults
    default_provider: str = Field(default="openrouter", alias="DEFAULT_PROVIDER")
    default_model: str = Field(default="openai/gpt-oss-20b:free", alias="DEFAULT_MODEL")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    # Privacy
    enable_redaction: bool = Field(default=True, alias="ENABLE_REDACTION")
    quiet_logs: bool = Field(default=True, alias="QUIET_LOGS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
