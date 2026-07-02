from qdrant_client import QdrantClient

from .config import settings


def create_qdrant_client() -> QdrantClient:
    if settings.QDRANT_URL:
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        https=settings.QDRANT_HTTPS,
        api_key=settings.QDRANT_API_KEY,
    )
