"""Pandas backend for dfmacro expression compilation."""

from __future__ import annotations

import difflib
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
            available = list(df.columns)
            matches = difflib.get_close_matches(expr.name, available, n=1, cutoff=0.6)
            suggestion = f" Did you mean '{matches[0]}'?" if matches else ""

            raise CompileError(
                f"Column {expr.name!r} not found. Available: {available!r}.{suggestion}"
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
            return acc.strip(expr.kwargs["chars"])
        elif expr.method == "lstrip":
            return acc.lstrip(expr.kwargs["chars"])
        elif expr.method == "rstrip":
            return acc.rstrip(expr.kwargs["chars"])
        elif expr.method == "len":
            return acc.len()
        elif expr.method == "replace":
            return acc.replace(
                expr.kwargs["pat"],
                expr.kwargs["repl"],
                regex=expr.kwargs["regex"],
            )
        elif expr.method == "split":
            return acc.split(expr.kwargs["pat"])
        elif expr.method == "slice":
            start = expr.kwargs["start"]
            length = expr.kwargs["length"]
            stop = None if length is None else (start or 0) + length
            return acc.slice(start, stop)
        elif expr.method == "pad":
            return acc.pad(
                expr.kwargs["width"],
                side=expr.kwargs["side"],
                fillchar=expr.kwargs["fillchar"],
            )
        elif expr.method == "zfill":
            return acc.zfill(expr.kwargs["width"])
        elif expr.method == "match":
            return acc.match(expr.kwargs["pat"])
        elif expr.method == "extract":
            return acc.extract(expr.kwargs["pat"], expand=True).iloc[
                :, expr.kwargs["group"] - 1
            ]
        elif expr.method == "title":
            return acc.title()
        elif expr.method == "count":
            return acc.count(expr.kwargs["pat"])
        elif expr.method == "reverse":
            return acc[::-1]
        elif expr.method == "fullmatch":
            return acc.fullmatch(expr.kwargs["pat"])
        elif expr.method == "isalnum":
            return acc.isalnum()
        elif expr.method == "isalpha":
            return acc.isalpha()
        elif expr.method == "isdigit":
            return acc.isdigit()
        elif expr.method == "islower":
            return acc.islower()
        elif expr.method == "isupper":
            return acc.isupper()
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
        elif expr.part == "second":
            return acc.second
        elif expr.part == "microsecond":
            return acc.microsecond
        elif expr.part == "nanosecond":
            return acc.nanosecond
        elif expr.part == "dayofweek":
            return acc.day_of_week
        elif expr.part == "quarter":
            return acc.quarter
        elif expr.part == "dayofyear":
            return acc.day_of_year
        elif expr.part == "week":
            return acc.isocalendar().week
        elif expr.part == "days_in_month":
            return acc.days_in_month
        elif expr.part == "is_month_start":
            return acc.is_month_start
        elif expr.part == "is_month_end":
            return acc.is_month_end
        elif expr.part == "is_year_start":
            return acc.is_year_start
        elif expr.part == "is_year_end":
            return acc.is_year_end
        elif expr.part == "is_leap_year":
            return acc.is_leap_year
        elif expr.part == "month_name":
            return acc.month_name()
        elif expr.part == "day_name":
            return acc.day_name()
        elif expr.part == "normalize":
            return acc.normalize()
        elif expr.part == "floor":
            return acc.floor(expr.kwargs["freq"])
        elif expr.part == "ceil":
            return acc.ceil(expr.kwargs["freq"])
        elif expr.part == "round":
            return acc.round(expr.kwargs["freq"])
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
