"""In-memory conversation history.

A TTL-bounded dict of ``session_id -> recent messages``. Good enough for a
single-process deployment; swap for Redis when running multiple workers
(the class intentionally mirrors a dict-like interface to make that swap easy).
"""

from __future__ import annotations

import time
import uuid
from collections import deque

from rag.llm import Message


class SessionStore:
    def __init__(self, max_turns: int = 6, ttl_seconds: int = 3600) -> None:
        self._maxlen = max_turns * 2  # user + assistant messages
        self._ttl = ttl_seconds
        self._sessions: dict[str, dict] = {}

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def _gc(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now - v["touched"] > self._ttl]
        for k in expired:
            del self._sessions[k]

    def get(self, session_id: str) -> list[Message]:
        self._gc()
        entry = self._sessions.get(session_id)
        if entry is None:
            return []
        entry["touched"] = time.time()
        return list(entry["messages"])

    def append(self, session_id: str, role: str, content: str) -> None:
        self._gc()
        entry = self._sessions.setdefault(
            session_id, {"messages": deque(maxlen=self._maxlen), "touched": time.time()}
        )
        entry["messages"].append({"role": role, "content": content})
        entry["touched"] = time.time()

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
