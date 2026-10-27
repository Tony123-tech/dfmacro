"""Polars backend for dfmacro expression compilation."""

from __future__ import annotations

import difflib
from typing import Any, Callable

import polars as pl

from ..expr import (
    AccessorExpr,
    BinaryExpr,
    ColumnExpr,
    Expr,
    Literal,
    MathExpr,
    UnaryExpr,
    WhenExpr,
    WindowExpr,
)
from . import CompileError
from .accessor import compile_accessor_method

_BINARY_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
}


def compile_expr(expr: Expr, df: Any) -> Any:
    if isinstance(expr, AccessorExpr):
        col = compile_expr(expr.column, df)
        return compile_accessor_method(expr, col, "polars")

    if isinstance(expr, ColumnExpr):
        if expr.name not in df.columns:
            available = df.columns
            matches = difflib.get_close_matches(expr.name, available, n=1, cutoff=0.6)
            suggestion = f" Did you mean '{matches[0]}'?" if matches else ""

            raise CompileError(
                f"Column {expr.name!r} not found. Available: {available}.{suggestion}"
            )
        return pl.col(expr.name)

    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, BinaryExpr):
        left = compile_expr(expr.left, df)
        right = compile_expr(expr.right, df)
        op_func = _BINARY_OPS.get(expr.op)
        if op_func is None:
            raise CompileError(f"Unsupported operator: {expr.op!r}")
        try:
            return op_func(left, right)
        except Exception as exc:
            raise CompileError(f"Failed to apply {expr.op!r}: {exc}") from exc

    if isinstance(expr, UnaryExpr):
        operand = compile_expr(expr.operand, df)
        if expr.op == "~":
            return ~operand
        raise CompileError(f"Unsupported unary op: {expr.op!r}")

    if isinstance(expr, WhenExpr):
        cond = compile_expr(expr.condition, df)
        then_val = compile_expr(expr.then_value, df)
        else_val = (
            compile_expr(expr.else_value, df) if expr.else_value is not None else None
        )
        if else_val is not None:
            return pl.when(cond).then(then_val).otherwise(else_val)
        return pl.when(cond).then(then_val)

    if isinstance(expr, WindowExpr):
        col = compile_expr(expr.column, df)
        if expr.func == "rank":
            return col.rank(
                method=expr.kwargs["method"], descending=expr.kwargs["descending"]
            )
        elif expr.func == "cumsum":
            return col.cum_sum()
        elif expr.func == "cummax":
            return col.cum_max()
        elif expr.func == "cummin":
            return col.cum_min()
        elif expr.func == "shift":
            return col.shift(expr.kwargs["periods"])
        elif expr.func == "diff":
            return col.diff()
        elif expr.func == "pct_change":
            return col.pct_change()
        raise CompileError(f"Unsupported window func: {expr.func}")

    if isinstance(expr, MathExpr):
        col = compile_expr(expr.column, df)
        if expr.func == "abs":
            return col.abs()
        elif expr.func == "round":
            return col.round(expr.kwargs["n"])
        elif expr.func == "sqrt":
            return col.sqrt()
        elif expr.func == "pow":
            return col.pow(expr.kwargs["n"])
        raise CompileError(f"Unsupported math func: {expr.func}")

    raise CompileError(f"Unknown node: {type(expr).__name__}")
