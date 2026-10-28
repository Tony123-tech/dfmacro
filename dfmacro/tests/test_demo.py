"""Tests for the safe Gradio demo query parser and runner."""

import pandas as pd
import pytest

import app


def test_query_parser_accepts_default_filter():
    expression = app.parse_query("df[price > 100]", list(app._DATA.columns))

    assert repr(expression) == "(col('price') > lit(100))"


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('echo unsafe')",
        "df[price > 100]; print('unsafe')",
        "other[price > 100]",
        "df[unknown_column > 1]",
        "df[price > 1 > 0]",
    ],
)
def test_query_parser_rejects_unsupported_or_executable_input(source):
    with pytest.raises(ValueError):
        app.parse_query(source, list(app._DATA.columns))


@pytest.mark.parametrize("backend", ["Pandas", "Polars", "DuckDB"])
def test_demo_query_runs_each_backend(backend):
    result, status = app.run_query("df[price > 100]", backend)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert (result["price"] > 100).all()
    assert status.startswith("Completed in")


def test_demo_returns_validation_error_without_executing_code():
    result, status = app.run_query(
        "__import__('os').system('echo unsafe')", "Pandas"
    )

    assert result.empty
    assert status.startswith("Query failed")
    assert "Enter one filter expression" in status
