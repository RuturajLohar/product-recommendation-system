import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "secretpassword")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "recommender_db")
    
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "rec_qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    
    REDIS_HOST: str = os.getenv("REDIS_HOST", "rec_redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))

settings = Settings()
