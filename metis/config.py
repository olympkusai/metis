from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl, validator
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(..., description="OpenAI API Key")
    database_dsn: str = Field(default="", description="Postgres DSN (legacy — no longer used for market data)")
    conversation_database_url: str = Field(..., description="Postgres DSN for db-metis (conversations, chat_messages)")
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL (currently unused — no code path consumes it)")
    api_base_url: HttpUrl = Field(default="https://api.olympkusai.com/api/v1", description="API base URL")
    pluto_base_url: str = Field(default="https://api.olympkusai.com/pluto/api/v1", description="Pluto personal-finance API base URL (via Nike gateway)")
    pluto_request_timeout_seconds: float = Field(default=15.0, description="Timeout in seconds for Pluto API requests")
    soter_base_url: str = Field(default="https://api.olympkusai.com/soter", description="Soter auth/identity service base URL")
    soter_request_timeout_seconds: float = Field(default=10.0, description="Timeout in seconds for Soter API requests")
    hermes_base_url: str = Field(default="https://api.olympkusai.com/hermes/mcp", description="Hermes MCP server URL (via Nike gateway)")
    hermes_request_timeout_seconds: float = Field(default=30.0, description="Timeout in seconds for Hermes MCP requests")
    soter_jwks_url: str = Field(default="https://api.olympkusai.com/soter/.well-known/jwks.json", description="Soter JWKS endpoint for RS256 token validation")
    oidc_issuer: str = Field(default="https://auth.olympkusai.com", description="Expected JWT issuer (must match Soter's OIDC_ISSUER)")
    jwks_cache_ttl_seconds: float = Field(default=600.0, description="TTL in seconds for cached JWKS keys")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent API requests")
    agent_version: str = Field(default="v1", description="Agent graph version: 'v1' (fixed pipeline) or 'v2' (ReAct loop)")

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
