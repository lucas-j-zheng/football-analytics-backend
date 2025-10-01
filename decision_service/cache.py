from typing import Optional
from cachetools import LRUCache
from hashlib import blake2b
import json


class SimpleLRUCache:
    def __init__(self, maxsize: int = 100_000):
        self.cache = LRUCache(maxsize=maxsize)

    @staticmethod
    def _key(obj: dict) -> str:
        canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return blake2b(canonical.encode(), digest_size=16).hexdigest()

    def get(self, obj: dict) -> Optional[dict]:
        return self.cache.get(self._key(obj))

    def set(self, obj: dict, value: dict) -> None:
        self.cache[self._key(obj)] = value


cache = SimpleLRUCache(maxsize=200_000)


