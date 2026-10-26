"""Polars backend for dfmacro expression compilation."""

from __future__ import annotations

import difflib
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
            return col.str.strip_chars(expr.kwargs["chars"])
        elif expr.method == "lstrip":
            return col.str.strip_chars_start(expr.kwargs["chars"])
        elif expr.method == "rstrip":
            return col.str.strip_chars_end(expr.kwargs["chars"])
        elif expr.method == "len":
            return col.str.len_chars()
        elif expr.method == "replace":
            return col.str.replace(
                expr.kwargs["pat"],
                expr.kwargs["repl"],
                literal=not expr.kwargs["regex"],
                n=-1,
            )
        elif expr.method == "split":
            return col.str.split(expr.kwargs["pat"])
        elif expr.method == "slice":
            return col.str.slice(
                expr.kwargs["start"] or 0, expr.kwargs["length"]
            )
        elif expr.method == "pad":
            width = expr.kwargs["width"]
            side = expr.kwargs["side"]
            fillchar = expr.kwargs["fillchar"]
            if side == "left":
                return col.str.pad_start(width, fill_char=fillchar)
            if side == "right":
                return col.str.pad_end(width, fill_char=fillchar)
            length = col.str.len_chars()
            left_width = (width + length) // 2
            return (
                pl.when(length < width)
                .then(
                    col.str.pad_start(left_width, fill_char=fillchar).str.pad_end(
                        width, fill_char=fillchar
                    )
                )
                .otherwise(col)
            )
        elif expr.method == "zfill":
            return col.str.zfill(expr.kwargs["width"])
        elif expr.method == "match":
            return col.str.contains(f"^(?:{expr.kwargs['pat']})")
        elif expr.method == "extract":
            return col.str.extract(
                expr.kwargs["pat"], group_index=expr.kwargs["group"]
            )
        elif expr.method == "title":
            return col.str.to_titlecase()
        elif expr.method == "count":
            return col.str.count_matches(expr.kwargs["pat"])
        elif expr.method == "reverse":
            return col.str.reverse()
        elif expr.method == "fullmatch":
            return col.str.contains(f"^(?:{expr.kwargs['pat']})$")
        elif expr.method == "isalnum":
            return col.str.contains(r"^[\p{L}\p{N}]+$")
        elif expr.method == "isalpha":
            return col.str.contains(r"^\p{L}+$")
        elif expr.method == "isdigit":
            return col.str.contains(r"^\p{Nd}+$")
        elif expr.method == "islower":
            return col.str.contains(r"\p{Ll}") & (
                col.str.to_lowercase() == col
            )
        elif expr.method == "isupper":
            return col.str.contains(r"\p{Lu}") & (
                col.str.to_uppercase() == col
            )
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
        elif expr.part == "second":
            return col.dt.second()
        elif expr.part == "microsecond":
            return col.dt.microsecond()
        elif expr.part == "nanosecond":
            return col.dt.nanosecond() % 1000
        elif expr.part == "dayofweek":
            return col.dt.weekday() - 1
        elif expr.part == "quarter":
            return col.dt.quarter()
        elif expr.part == "dayofyear":
            return col.dt.ordinal_day()
        elif expr.part == "week":
            return col.dt.week()
        elif expr.part == "days_in_month":
            return col.dt.month_end().dt.day()
        elif expr.part == "is_month_start":
            return col.dt.day() == 1
        elif expr.part == "is_month_end":
            return col == col.dt.month_end()
        elif expr.part == "is_year_start":
            return (col.dt.month() == 1) & (col.dt.day() == 1)
        elif expr.part == "is_year_end":
            return (col.dt.month() == 12) & (col.dt.day() == 31)
        elif expr.part == "is_leap_year":
            return col.dt.is_leap_year()
        elif expr.part == "month_name":
            return col.dt.strftime("%B")
        elif expr.part == "day_name":
            return col.dt.strftime("%A")
        elif expr.part == "normalize":
            return col.dt.truncate("1d")
        elif expr.part == "floor":
            return col.dt.truncate(expr.kwargs["freq"])
        elif expr.part == "ceil":
            truncated = col.dt.truncate(expr.kwargs["freq"])
            return pl.when(col == truncated).then(truncated).otherwise(
                truncated.dt.offset_by(expr.kwargs["freq"])
            )
        elif expr.part == "round":
            return col.dt.round(expr.kwargs["freq"])
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
