from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    mlflow_tracking_uri: str
    database_url: str
    redis_url: str
    chromadb_persist_directory: str

    class Config:
        env_file = ".env"

settings = Settings()
