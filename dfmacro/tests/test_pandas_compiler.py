"""Tests for the pandas backend compiler."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from dfmacro.compiler import CompileError, compile_expr
from dfmacro.expr import BinaryExpr, ColumnExpr, Literal


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
