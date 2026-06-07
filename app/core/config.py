from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Reco Server"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    smooth_frames: int = 8
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_enabled: bool = True

    @property
    def cors_origin_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
