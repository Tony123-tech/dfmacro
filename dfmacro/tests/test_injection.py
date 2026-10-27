"""Regression tests ensuring column resolution never mutates process tracing."""

from __future__ import annotations

import sys

import pandas as pd

from dfmacro import enable


def test_decorator_does_not_install_or_leave_a_trace_hook(monkeypatch):
    frame = pd.DataFrame({"price": [50, 150]})

    @enable(frame)
    def query():
        return df[price > 100]  # noqa: F821

    def reject_trace_installation(_trace):
        raise AssertionError("dfmacro must not install a trace hook")

    monkeypatch.setattr(sys, "settrace", reject_trace_installation)
    trace_before = sys.gettrace()
    result = query()

    assert result.raw["price"].tolist() == [150]
    assert sys.gettrace() is trace_before
    assert sys.gettrace() is None


def test_decorator_does_not_mutate_module_globals():
    frame = pd.DataFrame({"injection_test_column": [1, 2]})

    @enable(frame)
    def query():
        return df[injection_test_column > 1]  # noqa: F821

    query()

    assert "injection_test_column" not in globals()
    assert "df" not in globals()
