"""Application settings loaded from environment variables via pydantic-settings.

Replaces the previous dotenv + plain class approach with type-safe
validated configuration.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Keboli Interview Agent.

    All values are loaded from environment variables or a `.env` file.
    """

    GROQ_API_KEY: str | None = None
    MAIN_BACKEND_URL: str = "http://localhost:8000"
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    LIVEKIT_URL: str | None = None
    LIVEKIT_API_KEY: str | None = None
    LIVEKIT_API_SECRET: str | None = None

    DEEPGRAM_API_KEY: str | None = None
    BEY_API_KEY: str | None = None

    DEBUG_LOG_PATH: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

config = settings
