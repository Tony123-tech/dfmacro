"""Acceptance tests for the injection layer.

Covers the five cases agreed during design:
1. Basic injection visible inside the function body.
2. Caller frame is not polluted after return.
3. Nested function does not inherit injection.
4. Exception paths still clean up.
5. verify() reports failed names and error message carries diagnostics.

Note: as of Python 3.14, bare names in a function body compile to
LOAD_GLOBAL, so GlobalInjection (frame.f_globals) is the primary
strategy. Tests assert against globals() accordingly.
"""

from __future__ import annotations

import sys

import pytest

from dfmacro.injection import (
    GlobalInjection,
    InjectionError,
    get_strategy,
    inject_or_raise,
)


class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"FakeColumn({self.name!r})"


def _install_tracer_for(func, mapping):
    """Minimal tracer harness: inject at call, cleanup at return/exception."""
    strategy = get_strategy()
    names = list(mapping.keys())

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code is func.__code__:
            inject_or_raise(strategy, frame, mapping)
            return _make_local_tracer(strategy, frame, names)
        return tracer

    def _make_local_tracer(strategy, target_frame, names):
        def local_tracer(frame, event, arg):
            if frame is target_frame and event in ("return", "exception"):
                strategy.cleanup(frame, names)
            return local_tracer

        return local_tracer

    return tracer


def test_basic_injection_visible_in_body():
    """case 1: injected name is readable inside the function body."""
    mapping = {"price": FakeColumn("price")}

    def target():
        return price

    tracer = _install_tracer_for(target, mapping)
    sys.settrace(tracer)
    try:
        result = target()
    finally:
        sys.settrace(None)

    assert isinstance(result, FakeColumn)
    assert result.name == "price"


def test_no_leak_into_caller_frame():
    """case 1b: injected name removed from globals after return."""
    mapping = {"price": FakeColumn("price")}

    def target():
        return price

    tracer = _install_tracer_for(target, mapping)
    sys.settrace(tracer)
    try:
        target()
        leaked = "price" in globals()
    finally:
        sys.settrace(None)

    assert not leaked


def test_nested_function_does_not_leak():
    """case 2: inner function is not injected by the outer tracer.

    The tracer only injects when the call event matches the target
    function's code object. The inner function has a different code
    object, so it must NOT see 'price'.
    """
    mapping = {"price": FakeColumn("price")}

    def target():
        def inner():
            return price

        return inner

    tracer = _install_tracer_for(target, mapping)
    sys.settrace(tracer)
    try:
        inner = target()
    finally:
        sys.settrace(None)

    with pytest.raises(NameError):
        inner()


def test_exception_path_cleans_up():
    """case 3: injected names removed even when body raises."""
    mapping = {"price": FakeColumn("price")}

    def target():
        raise RuntimeError("boom")

    tracer = _install_tracer_for(target, mapping)
    sys.settrace(tracer)
    try:
        with pytest.raises(RuntimeError):
            target()
        leaked = "price" in globals()
    finally:
        sys.settrace(None)

    assert not leaked


def test_verify_reports_failed_names():
    """case 4: verify() returns missing names; error message has diagnostics."""

    class BrokenStrategy(GlobalInjection):
        def verify(self, frame, names):
            return list(names)  # always report failure

    strategy = BrokenStrategy()
    mapping = {"price": FakeColumn("price")}

    class FakeFrame:
        def __init__(self) -> None:
            self.f_locals = {}
            self.f_globals = {}
            self.f_code = None

    frame = FakeFrame()
    failed = strategy.verify(frame, ["price"])
    assert failed == ["price"]

    with pytest.raises(InjectionError) as excinfo:
        inject_or_raise(strategy, frame, mapping)

    message = str(excinfo.value)
    assert "price" in message
    assert "GlobalInjection" in message
    assert sys.version in message
