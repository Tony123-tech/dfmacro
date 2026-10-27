"""The @enable decorator for source-based DataFrame expression rewriting."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

from .df_wrapper import DF
from .dynamic import compile_dynamic

F = TypeVar("F", bound=Callable[..., Any])
_DF_NAMES = ("df", "data", "frame", "dataframe")


def _get_frame_name(func: Callable[..., Any], df: Any) -> str:
    signature = inspect.signature(func)
    parameters = signature.parameters
    for name in _DF_NAMES:
        if name in parameters:
            return name
    for name, value in func.__globals__.items():
        if value is df:
            return name
    for name in _DF_NAMES:
        if name in func.__code__.co_freevars:
            return name
    return "df"


def enable(df: Any) -> Callable[[F], F]:
    """Decorate a function so bare DataFrame column names become expressions."""
    wrapped = df if isinstance(df, DF) else DF(df)
    columns = list(wrapped.columns)

    def decorator(func: F) -> F:
        frame_name = _get_frame_name(func, df)
        generated = compile_dynamic(
            func,
            "__dfmacro_columns__",
            columns,
            frame_name=frame_name,
        )
        signature = inspect.signature(func)
        columns_argument = generated.__dfmacro_columns_argument__
        frame_argument = generated.__dfmacro_frame_argument__

        def invoke(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            if frame_name in bound.arguments:
                bound.arguments[frame_name] = wrapped
            call_kwargs = dict(bound.kwargs)
            call_kwargs[columns_argument] = generated.__dfmacro_columns_factory__()
            call_kwargs[frame_argument] = wrapped
            return generated(*bound.args, **call_kwargs)

        if inspect.iscoroutinefunction(func):
            async def async_invoke(*args: Any, **kwargs: Any) -> Any:
                bound = signature.bind(*args, **kwargs)
                if frame_name in bound.arguments:
                    bound.arguments[frame_name] = wrapped
                call_kwargs = dict(bound.kwargs)
                call_kwargs[columns_argument] = generated.__dfmacro_columns_factory__()
                call_kwargs[frame_argument] = wrapped
                return await generated(*bound.args, **call_kwargs)

            result = functools.wraps(func)(async_invoke)
        else:
            result = functools.wraps(func)(invoke)
        return cast(F, result)

    return decorator
