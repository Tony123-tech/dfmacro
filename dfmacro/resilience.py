"""Retry helpers for transient backend failures."""

from __future__ import annotations

import functools
import logging
import math
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])
_LOGGER = logging.getLogger("dfmacro.resilience")


def retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError),
) -> Callable[[F], F]:
    """Retry a callable after selected failures using exponential backoff.

    ``max_retries`` counts retries after the initial call. By default, only
    timeout and connection errors are retried; deterministic failures should
    be surfaced immediately. The wait before retry number ``n`` is
    ``backoff_factor ** n`` seconds.
    """
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError("max_retries must be a non-negative integer.")
    if (
        isinstance(backoff_factor, bool)
        or not isinstance(backoff_factor, (int, float))
        or not math.isfinite(backoff_factor)
        or backoff_factor < 0
    ):
        raise ValueError("backoff_factor must be a finite, non-negative number.")
    if not isinstance(exceptions, tuple) or not exceptions or not all(
        isinstance(exception, type) and issubclass(exception, BaseException)
        for exception in exceptions
    ):
        raise TypeError("exceptions must be a non-empty tuple of exception types.")

    def decorate(function: F) -> F:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return function(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_retries:
                        raise
                    delay = backoff_factor ** (attempt + 1)
                    _LOGGER.warning(
                        "%s failed (attempt %d/%d): %s; retrying in %.3f seconds",
                        function.__qualname__,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise RuntimeError("Retry loop exited unexpectedly.")

        return cast(F, wrapped)

    return decorate
