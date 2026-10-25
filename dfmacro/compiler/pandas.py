"""Pandas backend for dfmacro expression compilation."""

from __future__ import annotations

from typing import Any, Callable

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
                f"Column {expr.name!r} not found. Available: {list(df.columns)!r}"
            )
        return df[expr.name]
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, BinaryExpr):
        left = compile_expr(expr.left, df)
        right = compile_expr(expr.right, df)
        op_func = _BINARY_OPS.get(expr.op)
        if op_func is None:
            raise CompileError(f"Unsupported binary operator: {expr.op!r}")
        try:
            return op_func(left, right)
        except Exception as exc:
            raise CompileError(f"Failed to apply operator {expr.op!r}: {exc}") from exc
    if isinstance(expr, UnaryExpr):
        operand = compile_expr(expr.operand, df)
        if expr.op == "~":
            return ~operand
        raise CompileError(f"Unsupported unary operator: {expr.op!r}")
    if isinstance(expr, StringMethodExpr):
        col = compile_expr(expr.column, df)
        acc = col.str
        if expr.method == "contains":
            return acc.contains(
                expr.kwargs["pat"], regex=expr.kwargs.get("regex", False)
            )
        elif expr.method == "startswith":
            return acc.startswith(expr.kwargs["pat"])
        elif expr.method == "endswith":
            return acc.endswith(expr.kwargs["pat"])
        elif expr.method == "lower":
            return acc.lower()
        elif expr.method == "upper":
            return acc.upper()
        elif expr.method == "strip":
            return acc.strip()
        elif expr.method == "len":
            return acc.len()
        elif expr.method == "replace":
            return acc.replace(expr.kwargs["pat"], expr.kwargs["repl"])
        elif expr.method == "split":
            return acc.split(expr.kwargs["pat"])
        elif expr.method == "slice":
            return acc.slice(expr.kwargs["start"], expr.kwargs["length"])
        elif expr.method == "pad":
            return acc.pad(expr.kwargs["width"], side=expr.kwargs["side"])
        elif expr.method == "zfill":
            return acc.zfill(expr.kwargs["width"])
        elif expr.method == "match":
            return acc.match(expr.kwargs["pat"])
        elif expr.method == "extract":
            return acc.extract(expr.kwargs["pat"])
        raise CompileError(f"Unsupported string method: {expr.method}")
    if isinstance(expr, DatePartExpr):
        col = compile_expr(expr.column, df)
        acc = col.dt
        if expr.part == "year":
            return acc.year
        elif expr.part == "month":
            return acc.month
        elif expr.part == "day":
            return acc.day
        elif expr.part == "hour":
            return acc.hour
        elif expr.part == "minute":
            return acc.minute
        elif expr.part == "dayofweek":
            return acc.dayofweek
        elif expr.part == "quarter":
            return acc.quarter
        elif expr.part == "strftime":
            return acc.strftime(expr.kwargs["fmt"])
        raise CompileError(f"Unsupported date part: {expr.part}")
    if isinstance(expr, WhenExpr):
        import numpy as np

        cond = compile_expr(expr.condition, df)
        then_val = compile_expr(expr.then_value, df)
        else_val = (
            compile_expr(expr.else_value, df) if expr.else_value is not None else np.nan
        )
        return np.where(cond, then_val, else_val)
    if isinstance(expr, WindowExpr):
        col = compile_expr(expr.column, df)
        if expr.func == "rank":
            return col.rank(
                method=expr.kwargs["method"], ascending=not expr.kwargs["descending"]
            )
        elif expr.func == "cumsum":
            return col.cumsum()
        elif expr.func == "cummax":
            return col.cummax()
        elif expr.func == "cummin":
            return col.cummin()
        elif expr.func == "shift":
            return col.shift(periods=expr.kwargs["periods"])
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
            return col.pow(0.5)
        elif expr.func == "pow":
            return col.pow(expr.kwargs["n"])
        raise CompileError(f"Unsupported math func: {expr.func}")
    raise CompileError(f"Unknown expression node type: {type(expr).__name__}")
