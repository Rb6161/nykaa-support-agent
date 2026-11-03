"""In-memory response cache keyed by normalised query text."""
from __future__ import annotations

import re
import threading
from typing import Callable, Tuple


def normalize_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s-]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


class ResponseCache:
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, query: str, compute: Callable[[], dict], namespace: str = "") -> Tuple[dict, bool]:
        key = f"{namespace}::{normalize_query(query)}"
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key], True
        value = compute()  # computed outside the lock: a slow compute must not block other keys
        with self._lock:
            self.misses += 1
            self._store[key] = value
        return value, False

    def clear(self):
        with self._lock:
            self._store.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._store)}
