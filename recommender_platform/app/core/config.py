from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: str = "development"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = (
        "http://localhost:5173,http://localhost:5174,http://localhost:5175,"
        "http://localhost:5176,http://localhost:5177,http://127.0.0.1:5173,"
        "http://127.0.0.1:5174,http://127.0.0.1:5175,http://127.0.0.1:5176,"
        "http://127.0.0.1:5177"
    )

    DATABASE_URL: str | None = None
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = "change_me_local_password"
    POSTGRES_DB: str = "recommender_db"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_HTTPS: bool = False
    QDRANT_COLLECTION: str = "products"

    REDIS_ENABLED: bool = True
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
    RANKER_MODEL_PATH: str | None = None

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]

settings = Settings()
