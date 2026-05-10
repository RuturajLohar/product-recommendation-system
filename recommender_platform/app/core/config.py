from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: str = "development"

    DATABASE_URL: str | None = None
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = "secretpassword"
    POSTGRES_DB: str = "recommender_db"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    SBERT_MODEL: str = "all-MiniLM-L6-v2"
    INGEST_SAMPLE_SIZE: int = 25000
    INGEST_BATCH_SIZE: int = 128
    CACHE_TTL: int = 120

    # LLM Settings
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    USE_LLM_RERANKING: bool = False

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

settings = Settings()
