import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str

    # Redis / cache
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    CACHE_TTL_HOURS: int = 24

settings = Settings()