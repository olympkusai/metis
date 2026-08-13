"""In-memory rate limiter for the Metis chat endpoints.

Limits each user (identified by JWT `user_id`) to a configurable number of
messages per sliding time window. When the limit is exceeded, a
``RateLimitExceeded`` exception is raised which the caller (middleware or
dependency) translates into an HTTP 429 response with the body::

    {"error": "rate limit exceeded"}
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict


# Default limits — 20 messages per minute per user.
DEFAULT_MAX_REQUESTS = 20
DEFAULT_WINDOW_SECONDS = 60


class RateLimitExceeded(Exception):
    """Raised when a user has exceeded the allowed request rate."""


class RateLimiter:
    """Sliding-window in-memory rate limiter.

    Tracks request timestamps per user key and rejects requests once the
    configured number of requests fall inside the rolling window.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, user_id: str) -> None:
        """Record a request for *user_id* and raise if the limit is exceeded."""
        now = time.monotonic()
        window = self.window_seconds
        dq = self._hits[user_id]

        # Evict timestamps that fell out of the rolling window.
        while dq and now - dq[0] > window:
            dq.popleft()

        if len(dq) >= self.max_requests:
            raise RateLimitExceeded(
                f"rate limit exceeded for user {user_id}: "
                f"{self.max_requests} requests per {window}s"
            )

        dq.append(now)

    def reset(self, user_id: str | None = None) -> None:
        """Clear hit history for a single user (or all users when ``None``)."""
        if user_id is None:
            self._hits.clear()
        else:
            self._hits.pop(user_id, None)
