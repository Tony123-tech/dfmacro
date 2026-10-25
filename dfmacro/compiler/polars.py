"""Polars backend for dfmacro expression compilation."""

from __future__ import annotations

from typing import Any, Callable

import polars as pl

from ..expr import (
    BinaryExpr,
    ColumnExpr,
    DatePartExpr,
    Expr,
    Literal,
    MathExpr,
    StringMethodExpr,
    UnaryExpr,
    WhenExpr,
    WindowExpr,
)
from . import CompileError

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
    if isinstance(expr, ColumnExpr):
        if expr.name not in df.columns:
            raise CompileError(
                f"Column {expr.name!r} not found. Available: {df.columns}"
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
    if isinstance(expr, StringMethodExpr):
        col = compile_expr(expr.column, df)
        if expr.method == "contains":
            return col.str.contains(
                expr.kwargs["pat"], literal=not expr.kwargs.get("regex", False)
            )
        elif expr.method == "startswith":
            return col.str.starts_with(expr.kwargs["pat"])
        elif expr.method == "endswith":
            return col.str.ends_with(expr.kwargs["pat"])
        elif expr.method == "lower":
            return col.str.to_lowercase()
        elif expr.method == "upper":
            return col.str.to_uppercase()
        elif expr.method == "strip":
            return col.str.strip_chars()
        elif expr.method == "len":
            return col.str.len_chars()
        elif expr.method == "replace":
            return col.str.replace(expr.kwargs["pat"], expr.kwargs["repl"])
        elif expr.method == "split":
            return col.str.split(expr.kwargs["pat"])
        elif expr.method == "slice":
            return col.str.slice(expr.kwargs["start"], expr.kwargs["length"])
        elif expr.method == "pad":
            return col.str.pad(expr.kwargs["width"], side=expr.kwargs["side"])
        elif expr.method == "zfill":
            return col.str.zfill(expr.kwargs["width"])
        raise CompileError(f"Unsupported string method: {expr.method}")
    if isinstance(expr, DatePartExpr):
        col = compile_expr(expr.column, df)
        if expr.part == "year":
            return col.dt.year()
        elif expr.part == "month":
            return col.dt.month()
        elif expr.part == "day":
            return col.dt.day()
        elif expr.part == "hour":
            return col.dt.hour()
        elif expr.part == "minute":
            return col.dt.minute()
        elif expr.part == "dayofweek":
            return col.dt.weekday()
        elif expr.part == "quarter":
            return col.dt.quarter()
        elif expr.part == "strftime":
            return col.dt.strftime(expr.kwargs["fmt"])
        raise CompileError(f"Unsupported date part: {expr.part}")
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
