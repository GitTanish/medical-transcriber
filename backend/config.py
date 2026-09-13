from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    groq_api_key: str

    model_config = SettingsConfigDict(env_file=DOTENV_PATH, extra="ignore")


settings = Settings()