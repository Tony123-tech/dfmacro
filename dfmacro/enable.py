"""The @enable decorator: inject column names and bind DF wrapper."""

from __future__ import annotations

import builtins
import dis
import sys
from typing import Any, Callable, TypeVar

from .df_wrapper import DF
from .expr import ColumnExpr
from .injection import get_strategy, inject_or_raise

F = TypeVar("F", bound=Callable[..., Any])

_BUILTIN_NAMES = frozenset(dir(builtins))
_DF_NAMES = ("df", "data")

# Global registry to track dynamically added column names per frame.
# Key is id(frame), value is a list of injected names.
_DYNAMIC_NAMES: dict[int, list[str]] = {}


def _register_dynamic_name(frame: Any, name: str) -> None:
    frame_id = id(frame)
    if frame_id not in _DYNAMIC_NAMES:
        _DYNAMIC_NAMES[frame_id] = []
    if name not in _DYNAMIC_NAMES[frame_id]:
        _DYNAMIC_NAMES[frame_id].append(name)


def _scan_load_global_names(func: Callable[..., Any]) -> set[str]:
    names: set[str] = set()
    for instr in dis.get_instructions(func):
        if instr.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            arg = instr.argval
            if (
                isinstance(arg, str)
                and arg not in _BUILTIN_NAMES
                and not arg.isdigit()
                and not arg.startswith("_")
            ):
                names.add(arg)
    return names


def enable(df: Any) -> Callable[[F], F]:
    wrapped = DF(df)

    def decorator(func: F) -> F:
        strategy = get_strategy()
        loaded_names = _scan_load_global_names(func)

        # Pre-process closure cells if they contain our df.
        freevars = func.__code__.co_freevars
        closure = func.__closure__ or ()

        for name, cell in zip(freevars, closure):
            if name in _DF_NAMES:
                cell.cell_contents = wrapped

        def tracer(frame: Any, event: str, arg: Any) -> Any:
            if event == "call" and frame.f_code is func.__code__:
                local_map: dict[str, Any] = {}

                # 1. Inject missing column names
                existing = set(frame.f_globals) | set(frame.f_builtins)
                for name in loaded_names:
                    if name in existing:
                        continue
                    local_map[name] = ColumnExpr(name)

                # 2. Aggressively inject the DataFrame wrapper into the frame
                for df_name in _DF_NAMES:
                    local_map[df_name] = wrapped

                if local_map:
                    inject_or_raise(strategy, frame, local_map)

                return _make_local_tracer(strategy, frame, list(local_map))
            return tracer

        def _make_local_tracer(
            strategy: Any, target_frame: Any, injected_names: list[str]
        ) -> Any:
            def local_tracer(frame: Any, event: str, arg: Any) -> Any:
                if frame is target_frame and event in ("return", "exception"):
                    # Cleanup original injected names
                    if injected_names:
                        strategy.cleanup(frame, injected_names)

                    # Cleanup dynamic names
                    dynamic = _DYNAMIC_NAMES.pop(id(target_frame), [])
                    for name in dynamic:
                        frame.f_globals.pop(name, None)

                return local_tracer

            return local_tracer

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            old_trace = sys.gettrace()
            sys.settrace(tracer)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(old_trace)
                # Ensure cleanup even if tracer failed
                _DYNAMIC_NAMES.pop(id(sys._getframe(1)), None)

        return wrapper  # type: ignore[return-value]

    return decorator
