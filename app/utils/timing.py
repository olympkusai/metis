"""Timing utilities for performance monitoring."""

import time
from functools import wraps
from contextlib import contextmanager
from typing import Callable, Any, Generator, Optional


@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {name}: {elapsed:.4f}s")


def timed(name: Optional[str] = None) -> Callable:
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            timer_name = name or func.__name__
            print(f"[TIMER] {timer_name}: {elapsed:.4f}s")
            return result
        return wrapper
    return decorator


def timed_async(name: Optional[str] = None) -> Callable:
    """Decorator for timing async function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            timer_name = name or func.__name__
            print(f"[TIMER] {timer_name}: {elapsed:.4f}s")
            return result
        return wrapper
    return decorator
