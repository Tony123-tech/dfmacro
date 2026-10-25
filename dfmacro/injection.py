"""Locals injection strategies for dfmacro."""

from __future__ import annotations

import ctypes
import sys
from typing import Any, Mapping, Protocol

_MISSING = object()


class InjectionError(RuntimeError):
    """Raised when injection cannot be completed reliably."""


class InjectionStrategy(Protocol):
    def inject(self, frame: Any, mapping: Mapping[str, Any]) -> None: ...
    def cleanup(self, frame: Any, names: list[str]) -> None: ...
    def verify(self, frame: Any, names: list[str]) -> list[str]: ...
    def describe(self) -> str: ...


class GlobalInjection:
    """Primary strategy: inject into frame.f_globals.

    Tracks original values to prevent deleting pre-existing globals.
    """

    def __init__(self) -> None:
        # Store original values by frame id to avoid adding attributes to frames
        self._originals: dict[int, dict[str, Any]] = {}

    def inject(self, frame: Any, mapping: Mapping[str, Any]) -> None:
        frame_id = id(frame)
        if frame_id not in self._originals:
            self._originals[frame_id] = {}

        originals = self._originals[frame_id]

        for name, value in mapping.items():
            if name not in originals:
                # Save original value (or a sentinel if it didn't exist)
                originals[name] = frame.f_globals.get(name, _MISSING)

            # Overwrite the global
            frame.f_globals[name] = value

    def cleanup(self, frame: Any, names: list[str]) -> None:
        frame_id = id(frame)
        originals = self._originals.get(frame_id, {})

        for name in names:
            orig = originals.get(name, _MISSING)
            if orig is _MISSING:
                # It didn't exist before, safe to delete
                frame.f_globals.pop(name, None)
            else:
                # It existed before (e.g., the user's `df`), restore it
                frame.f_globals[name] = orig

        # Clean up the tracking dict
        if frame_id in self._originals:
            del self._originals[frame_id]

    def verify(self, frame: Any, names: list[str]) -> list[str]:
        g = frame.f_globals
        return [n for n in names if n not in g]

    def describe(self) -> str:
        return "GlobalInjection (frame.f_globals with restore)"


class ProxyInjection:
    def inject(self, frame: Any, mapping: Mapping[str, Any]) -> None:
        for name, value in mapping.items():
            frame.f_locals[name] = value

    def cleanup(self, frame: Any, names: list[str]) -> None:
        for name in names:
            frame.f_locals.pop(name, None)

    def verify(self, frame: Any, names: list[str]) -> list[str]:
        local_ns = frame.f_locals
        return [n for n in names if n not in local_ns]

    def describe(self) -> str:
        return "ProxyInjection (Python 3.13+ f_locals proxy, LOAD_FAST only)"


class CapiInjection:
    def _locals_to_fast(self, frame: Any) -> None:
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))

    def inject(self, frame: Any, mapping: Mapping[str, Any]) -> None:
        local_ns = frame.f_locals
        for name, value in mapping.items():
            local_ns[name] = value
        self._locals_to_fast(frame)

    def cleanup(self, frame: Any, names: list[str]) -> None:
        local_ns = frame.f_locals
        for name in names:
            local_ns.pop(name, None)
        self._locals_to_fast(frame)

    def verify(self, frame: Any, names: list[str]) -> list[str]:
        local_ns = frame.f_locals
        return [n for n in names if n not in local_ns]

    def describe(self) -> str:
        return (
            f"CapiInjection (Python {sys.version_info.major}.{sys.version_info.minor})"
        )


def get_strategy() -> InjectionStrategy:
    # Use GlobalInjection by default because bare names compile to LOAD_GLOBAL,
    # and we need the restore mechanism for global df variables.
    return GlobalInjection()


def inject_or_raise(
    strategy: InjectionStrategy, frame: Any, mapping: Mapping[str, Any]
) -> None:
    strategy.inject(frame, mapping)
    failed = strategy.verify(frame, list(mapping.keys()))
    if failed:
        raise InjectionError(
            f"Failed to inject names {failed!r} into frame. "
            f"Strategy: {strategy.describe()}. "
            f"Python: {sys.version}."
        )
