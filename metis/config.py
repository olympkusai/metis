from pydantic_settings import BaseSettings
from pydantic import Field, HttpUrl, validator
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: str = Field(..., description="OpenAI API Key")
    database_url: str = Field(default="postgres://postgres:BGxE9aWYJP5Ai7rhLkGeQUcnt8Y4hvnq3IM282m7OgEtKIF4QjmMUbIND07qCBR9@88.99.66.165:5432/k0s_prd?sslmode=require", description="External k0s PostgreSQL connection URL (market data cache — owned by another system, do not repoint)")
    conversation_database_url: str = Field(..., description="Postgres DSN for Metis's own database (db-metis) — conversations, chat_messages, notifications")
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL (currently unused — no code path consumes it)")
    api_base_url: HttpUrl = Field(default="https://api.k0s.app/api/v1", description="API base URL")
    apollo_base_url: str = Field(default="http://apollo.internal:8000", description="Apollo ML API base URL")
    apollo_prediction_lookback_days: int = Field(default=90, description="Lookback window in days for Apollo predictions")
    apollo_train_lookback_days: int = Field(default=365, description="Lookback window in days for Apollo model training")
    apollo_backtest_periods: int = Field(default=5, description="Number of periods to request in Apollo backtests")
    apollo_backtest_error_threshold_pct: float = Field(default=2.5, description="Maximum allowed fifth-period backtest error percentage")
    apollo_train_max_attempts: int = Field(default=3, description="Maximum Apollo retraining attempts")
    apollo_poll_interval_seconds: int = Field(default=10, description="Polling interval in seconds after Apollo training starts")
    apollo_train_timeout_seconds: int = Field(default=600, description="Maximum wait in seconds for Apollo training to become usable")
    apollo_confidence_threshold: float = Field(default=0.35, description="Minimum Apollo confidence to treat a forecast as actionable")
    apollo_mape_threshold: float = Field(default=5.0, description="Maximum Apollo MAPE to treat a forecast as actionable")
    pluto_base_url: str = Field(default="https://api.olympkusai.com/pluto/api/v1", description="Pluto personal-finance API base URL (via Nike gateway)")
    pluto_request_timeout_seconds: float = Field(default=15.0, description="Timeout in seconds for Pluto API requests")
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
