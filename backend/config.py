from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    groq_api_key: str = ""
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    asr_model: str = "whisper-large-v3-turbo"
    llm_model: str = "qwen/qwen3.8-27b"
    provider_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    asr_context_chars: int = Field(default=1200, ge=0, le=10000)
    asr_prompt_max_chars: int = Field(
        default=800,
        ge=0,
        le=10000,
        description=(
            "Hard ceiling for the Whisper 'prompt' parameter. Groq documents a "
            "224-token prompt limit; clinical dialogue averages roughly four "
            "characters per token, so 800 characters is reserved to stay inside it."
        ),
    )
    vad_threshold: float = Field(default=0.35, gt=0, lt=1)
    vad_min_speech_duration_ms: int = Field(default=150, ge=0)
    vad_min_silence_duration_ms: int = Field(default=250, ge=0)
    vad_speech_pad_ms: int = Field(default=180, ge=0)
    vad_min_segment_seconds: float = Field(default=0.18, gt=0, le=5)
    noise_reduction_strength: float = Field(default=0.5, ge=0, le=1)
    max_audio_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    max_audio_duration_seconds: float = Field(default=600.0, gt=0)
    max_input_sample_rate: int = Field(default=192_000, ge=8_000)
    max_transcript_chars: int = Field(default=100_000, ge=1)
    max_language_chars: int = Field(default=16, ge=2, le=32)

    model_config = SettingsConfigDict(env_file=DOTENV_PATH, extra="ignore")

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        if not origins:
            raise ValueError("cors_origins must contain at least one origin")
        return ",".join(origins)

    @property
    def allowed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def cors_allow_credentials(self) -> bool:
        return "*" not in self.allowed_cors_origins


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
