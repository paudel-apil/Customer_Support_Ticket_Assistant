from pydantic_settings import BaseSettings, SettingsConfigDict
from decouple import config
from sqlalchemy.ext.declarative import declarative_base


SECRET_KEY = config('SECRET_KEY')

class Settings(BaseSettings):
    PROJECT_NAME: str = "Support Ticket Intelligence Platform"
    DATABASE_URL: str = f"postgresql://postgres:{config('DB_PASSWORD')}@localhost:{config('DB_PORT')}/{config('DB_NAME')}"
    QDRANT_URL: str = "http://localhost:6333"

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"  
    )

settings = Settings()

Base = declarative_base()
