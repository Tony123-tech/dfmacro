"""Tests for the lazy expression tree."""

from __future__ import annotations

import pytest

from dfmacro.expr import (
    BinaryExpr,
    ColumnExpr,
    Literal,
    UnaryExpr,
    structural_eq,
)


def test_column_leaf():
    c = ColumnExpr("price")
    assert c.name == "price"
    assert repr(c) == "col('price')"


def test_literal_leaf():
    lit = Literal(100)
    assert lit.value == 100
    assert repr(lit) == "lit(100)"


def test_comparison_builds_tree():
    expr = ColumnExpr("price") > 100
    expected = BinaryExpr(">", ColumnExpr("price"), Literal(100))
    assert structural_eq(expr, expected)


def test_and_builds_tree():
    expr = (ColumnExpr("price") > 100) & (ColumnExpr("category") == "electronics")
    expected = BinaryExpr(
        "&",
        BinaryExpr(">", ColumnExpr("price"), Literal(100)),
        BinaryExpr("==", ColumnExpr("category"), Literal("electronics")),
    )
    assert structural_eq(expr, expected)


def test_invert_builds_unary():
    expr = ~(ColumnExpr("price") > 100)
    expected = UnaryExpr("~", BinaryExpr(">", ColumnExpr("price"), Literal(100)))
    assert structural_eq(expr, expected)


def test_bool_raises_clear_error():
    with pytest.raises(TypeError, match="cannot be used with 'and'/'or'"):
        bool(ColumnExpr("price") > 100)


def test_and_word_raises_via_bool():
    with pytest.raises(TypeError, match="cannot be used with 'and'/'or'"):
        _ = (ColumnExpr("price") > 100) and (ColumnExpr("category") == "x")


def test_precedence_trap_on_gt():
    with pytest.raises(TypeError, match="Operator precedence trap"):
        _ = ColumnExpr("price") > (100 & (ColumnExpr("category") == "x"))


def test_rand_builds_tree_not_trap():
    # 100 & expr is a plain bitwise-and operation, not a precedence trap.
    expr = 100 & ColumnExpr("price")
    expected = BinaryExpr("&", Literal(100), ColumnExpr("price"))
    assert structural_eq(expr, expected)


def test_tree_is_immutable():
    # Expr.__slots__ = () prevents subclasses from gaining a __dict__,
    # so arbitrary attributes cannot be attached.
    expr = ColumnExpr("price")
    with pytest.raises(AttributeError):
        expr.unknown_attr = 1  # type: ignore[attr-defined]


def test_repr_is_readable():
    expr = (ColumnExpr("price") > 100) & (ColumnExpr("category") == "electronics")
    text = repr(expr)
    assert "col('price')" in text
    assert "col('category')" in text
    assert ">" in text
    assert "==" in text
    assert "&" in text

def test_column_vs_column_comparison_is_allowed():
    expr = ColumnExpr("a") > ColumnExpr("b")
    expected = BinaryExpr(">", ColumnExpr("a"), ColumnExpr("b"))
    assert structural_eq(expr, expected)
