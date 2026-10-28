"""Tests for retries and extension registries."""

import pandas as pd
import pytest

from dfmacro import DF, register_accessor, register_backend
from dfmacro.compiler import compile_expr
from dfmacro.expr import ColumnExpr
from dfmacro.resilience import retry


def test_retry_retries_transient_failures(monkeypatch):
    sleeps = []
    calls = []
    monkeypatch.setattr("dfmacro.resilience.time.sleep", sleeps.append)

    @retry(max_retries=2, backoff_factor=0.25)
    def eventually_succeeds():
        calls.append(None)
        if len(calls) < 3:
            raise TimeoutError("temporary")
        return "ok"

    assert eventually_succeeds() == "ok"
    assert len(calls) == 3
    assert sleeps == [0.25, 0.0625]


def test_retry_reraises_after_maximum_attempts(monkeypatch):
    monkeypatch.setattr("dfmacro.resilience.time.sleep", lambda _: None)
    calls = []

    @retry(max_retries=1, backoff_factor=0)
    def always_fails():
        calls.append(None)
        raise TimeoutError("unavailable")

    with pytest.raises(TimeoutError, match="unavailable"):
        always_fails()
    assert len(calls) == 2


def test_retry_does_not_retry_deterministic_failures(monkeypatch):
    sleeps = []
    calls = []
    monkeypatch.setattr("dfmacro.resilience.time.sleep", sleeps.append)

    @retry(max_retries=3)
    def invalid_operation():
        calls.append(None)
        raise ValueError("invalid input")

    with pytest.raises(ValueError, match="invalid input"):
        invalid_operation()
    assert len(calls) == 1
    assert sleeps == []


def test_duckdb_execution_retries_connection_failures(monkeypatch):
    duckdb = pytest.importorskip("duckdb")

    from dfmacro.duckdb_wrapper import _execute_dataframe

    monkeypatch.setattr("dfmacro.resilience.time.sleep", lambda _: None)

    class QueryResult:
        def df(self):
            return "result"

    class Connection:
        calls = 0

        def execute(self, query):
            self.calls += 1
            if self.calls == 1:
                raise duckdb.ConnectionException("temporary disconnect")
            return QueryResult()

    connection = Connection()
    assert _execute_dataframe(connection, "SELECT 1") == "result"
    assert connection.calls == 2


def test_custom_accessor_registers_expression_and_compiles():
    @register_accessor("str.test_lowercase")
    def test_lowercase(series):
        return series.str.lower()

    frame = pd.DataFrame({"name": ["ADA", "Grace"]})
    expression = ColumnExpr("name").str.test_lowercase()

    result = DF(frame)[expression == "ada"].raw
    assert result["name"].tolist() == ["ADA"]


def test_custom_backend_compiler_dispatches_for_declared_backend():
    class ExampleFrame:
        __dfmacro_backend__ = "test_backend"

    @register_backend("test_backend")
    def compile_example(expr, frame):
        return (type(expr).__name__, type(frame).__name__)

    result = compile_expr(ColumnExpr("value"), ExampleFrame())
    assert result == ("ColumnExpr", "ExampleFrame")


def test_df_explain_includes_columns_and_dtypes():
    description = DF(pd.DataFrame({"count": [1, 2]})).explain()
    assert "DataFrame" in description
    assert "count: int" in description
