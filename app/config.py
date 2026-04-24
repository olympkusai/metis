from pydantic_settings import BaseSettings
from pydantic import Field, HttpUrl, validator
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: str = Field(..., description="OpenAI API Key")
    database_url: str = Field(default="postgres://postgres:BGxE9aWYJP5Ai7rhLkGeQUcnt8Y4hvnq3IM282m7OgEtKIF4QjmMUbIND07qCBR9@88.99.66.165:5432/k0s_prd?sslmode=require", description="PostgreSQL connection URL")
    redis_url: str = Field(..., description="Redis connection URL")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API Key")
    api_base_url: HttpUrl = Field(default="https://api.k0s.app/api/v1", description="API base URL")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent API requests")

    class Config:
        env_file = ".env"

    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if not v or not v.startswith('sk-'):
            raise ValueError('OPENAI_API_KEY must be a valid OpenAI API key starting with "sk-"')
        return v


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Lazy initialization of settings with error handling."""
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}. Please check your .env file.")
    return _settings


# For backward compatibility
settings = get_settings()
