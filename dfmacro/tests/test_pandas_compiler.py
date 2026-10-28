"""Tests for the pandas backend compiler."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from dfmacro.compiler import CompileError, compile_expr
from dfmacro.expr import BinaryExpr, ColumnExpr, Literal
import dfmacro.compiler.pandas as pandas_compiler


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "price": [50, 150, 200, 80],
            "category": ["books", "electronics", "electronics", "books"],
            "brand": ["a", "b", "b", "a"],
        }
    )


def test_column_leaf_compiles_to_series(df):
    result = compile_expr(ColumnExpr("price"), df)
    assert result.equals(df["price"])


def test_literal_leaf_compiles_to_value(df):
    result = compile_expr(Literal(100), df)
    assert result == 100


def test_comparison_compiles_to_boolean_series(df):
    expr = ColumnExpr("price") > 100
    result = compile_expr(expr, df)
    expected = df["price"] > 100
    assert result.equals(expected)


def test_and_compiles_to_boolean_series(df):
    expr = (ColumnExpr("price") > 100) & (ColumnExpr("category") == "electronics")
    result = compile_expr(expr, df)
    expected = (df["price"] > 100) & (df["category"] == "electronics")
    assert result.equals(expected)


def test_or_compiles_to_boolean_series(df):
    expr = (ColumnExpr("price") > 150) | (ColumnExpr("brand") == "a")
    result = compile_expr(expr, df)
    expected = (df["price"] > 150) | (df["brand"] == "a")
    assert result.equals(expected)


def test_invert_compiles_to_negated_mask(df):
    expr = ~(ColumnExpr("price") > 100)
    result = compile_expr(expr, df)
    expected = ~(df["price"] > 100)
    assert result.equals(expected)


def test_arithmetic_compiles_to_numeric_series(df):
    expr = ColumnExpr("price") * 2 + 10
    result = compile_expr(expr, df)
    expected = df["price"] * 2 + 10
    assert result.equals(expected)


def test_missing_column_raises(df):
    with pytest.raises(CompileError, match="not found"):
        compile_expr(ColumnExpr("nonexistent"), df)


def test_unsupported_operator_raises(df):
    # Hand-build a node with an op the compiler does not know.
    expr = BinaryExpr("**", ColumnExpr("price"), Literal(2))
    with pytest.raises(CompileError, match="Unsupported binary operator"):
        compile_expr(expr, df)


def test_non_expr_raises(df):
    with pytest.raises(CompileError, match="expects an Expr node"):
        compile_expr(100, df)  # type: ignore[arg-type]


def test_unsupported_dataframe_type_raises():
    class FakeDF:
        pass

    with pytest.raises(CompileError, match="Unsupported DataFrame type"):
        compile_expr(ColumnExpr("price"), FakeDF())


def test_typo_suggestion(df):
    # User typed 'prce' instead of 'price'
    expr = ColumnExpr("prce") > 100
    with pytest.raises(CompileError, match="Did you mean 'price'"):
        compile_expr(expr, df)


def test_string_accessor_rejects_numeric_column(df):
    expr = ColumnExpr("price").str.lower()

    with pytest.raises(CompileError, match="requires an object or string column"):
        compile_expr(expr, df)


def test_string_accessor_accepts_object_column(df):
    expr = ColumnExpr("category").str.lower()

    result = compile_expr(expr, df)

    assert result.equals(df["category"].str.lower())


def test_datetime_accessor_rejects_non_datetime_column(df):
    expr = ColumnExpr("price").dt.year

    with pytest.raises(CompileError, match="requires a datetime64 column"):
        compile_expr(expr, df)


def test_datetime_accessor_accepts_datetime_column():
    frame = pd.DataFrame({"created_at": pd.to_datetime(["2025-01-01", "2026-02-02"])})
    expr = ColumnExpr("created_at").dt.year

    result = compile_expr(expr, frame)

    assert result.equals(frame["created_at"].dt.year)


def test_arrow_table_helper_exports_column_without_pandas_numpy_conversion(df):
    pa = pytest.importorskip("pyarrow")

    table = pandas_compiler._arrow_table_for_column(df, "price")

    assert isinstance(table, pa.Table)
    assert table.column("price").to_pylist() == df["price"].tolist()


def test_rust_arrow_comparison_dispatch_avoids_to_numpy(df, monkeypatch):
    class FakeRust:
        @staticmethod
        def filter_compare_arrow(table, column_name, op, scalar):
            assert table == "arrow-table"
            assert column_name == "price"
            assert op == ">"
            assert scalar == 100.0
            return [False, True, True, False]

    monkeypatch.setattr(pandas_compiler, "HAS_RUST", True)
    monkeypatch.setattr(pandas_compiler, "dfmacro_rs", FakeRust())
    monkeypatch.setattr(
        pandas_compiler, "_arrow_table_for_column", lambda frame, name: "arrow-table"
    )
    monkeypatch.setattr(
        pd.Series,
        "to_numpy",
        lambda *args, **kwargs: pytest.fail("to_numpy used"),
    )

    result = pandas_compiler.compile_expr(ColumnExpr("price") > 100, df)

    assert result.tolist() == [False, True, True, False]


def test_rust_arrow_window_dispatch_avoids_to_numpy(df, monkeypatch):
    class FakeRust:
        @staticmethod
        def numeric_window_arrow(table, column_name, operation):
            assert table == "arrow-table"
            assert column_name == "price"
            assert operation == "cumsum"
            return [50, 200, 400, 480]

    monkeypatch.setattr(pandas_compiler, "HAS_RUST", True)
    monkeypatch.setattr(pandas_compiler, "dfmacro_rs", FakeRust())
    monkeypatch.setattr(
        pandas_compiler, "_arrow_table_for_column", lambda frame, name: "arrow-table"
    )
    monkeypatch.setattr(
        pd.Series,
        "to_numpy",
        lambda *args, **kwargs: pytest.fail("to_numpy used"),
    )

    result = pandas_compiler.compile_expr(ColumnExpr("price").cumsum(), df)

    assert result.tolist() == [50, 200, 400, 480]
