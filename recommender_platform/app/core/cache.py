import json
from functools import lru_cache
from typing import Any, Optional

from redis import Redis

from .config import settings


@lru_cache(maxsize=1)
def redis_client() -> Redis:
    return Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)


def get_json(key: str) -> Optional[Any]:
    val = redis_client().get(key)
    if not val:
        return None
    return json.loads(val)


def set_json(key: str, value: Any, ttl_seconds: int) -> None:
    redis_client().setex(key, ttl_seconds, json.dumps(value))

