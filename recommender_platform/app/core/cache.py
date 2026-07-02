import json
from functools import lru_cache
from typing import Any, Optional

from redis import Redis, RedisError

from .config import settings


@lru_cache(maxsize=1)
def redis_client() -> Optional[Redis]:
    if not settings.REDIS_ENABLED:
        return None
    return Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)


def get_json(key: str) -> Optional[Any]:
    client = redis_client()
    if client is None:
        return None
    try:
        val = client.get(key)
    except RedisError:
        return None
    if not val:
        return None
    return json.loads(val)


def set_json(key: str, value: Any, ttl_seconds: int) -> None:
    client = redis_client()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, json.dumps(value))
    except RedisError:
        return


def delete_pattern(pattern: str) -> int:
    client = redis_client()
    if client is None:
        return 0
    try:
        keys = list(client.scan_iter(match=pattern))
    except RedisError:
        return 0
    if not keys:
        return 0
    try:
        return int(client.delete(*keys))
    except RedisError:
        return 0
