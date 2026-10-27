"""Pandas backend for dfmacro expression compilation."""

from __future__ import annotations

import difflib
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import dfmacro_rs

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from ..expr import (
    AccessorExpr,
    AggExpr,
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

_FLIP_OP = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "==": "==", "!=": "!="}


def compile_expr(expr: Expr, df: Any) -> Any:
    if isinstance(expr, AccessorExpr):
        column = compile_expr(expr.column, df)
        return compile_accessor_method(expr, column, "pandas")

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
        # --- RUST ACCELERATION FOR COMPARISONS ---
        if HAS_RUST and expr.op in _FLIP_OP:
            try:
                left = compile_expr(expr.left, df)
                right = compile_expr(expr.right, df)

                # Case 1: Column OP Literal
                if isinstance(expr.left, ColumnExpr) and isinstance(
                    expr.right, Literal
                ):
                    arr = left.to_numpy(dtype=float)
                    mask = dfmacro_rs.filter_compare(arr, expr.op, scalar=float(right))
                    return pd.Series(mask, index=df.index)

                # Case 2: Literal OP Column
                elif isinstance(expr.right, ColumnExpr) and isinstance(
                    expr.left, Literal
                ):
                    arr = right.to_numpy(dtype=float)
                    mask = dfmacro_rs.filter_compare(
                        arr, _FLIP_OP[expr.op], scalar=float(left)
                    )
                    return pd.Series(mask, index=df.index)
            except Exception:
                pass  # Fallback to native Pandas if Rust fails

        # --- PANDAS FALLBACK ---
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

    if isinstance(expr, WhenExpr):
        cond = compile_expr(expr.condition, df)
        then_val = compile_expr(expr.then_value, df)
        else_val = (
            compile_expr(expr.else_value, df) if expr.else_value is not None else np.nan
        )
        return np.where(cond, then_val, else_val)

    if isinstance(expr, WindowExpr):
        col = compile_expr(expr.column, df)

        # --- RUST ACCELERATION FOR STATS ---
        if HAS_RUST:
            try:
                if expr.func == "var":
                    return dfmacro_rs.var_parallel(col.to_numpy(dtype=float))
                elif expr.func == "std":
                    return dfmacro_rs.std_parallel(col.to_numpy(dtype=float))
                elif expr.func == "cumsum":
                    return pd.Series(
                        dfmacro_rs.cumsum_parallel(col.to_numpy(dtype=float)),
                        index=df.index,
                    )
                elif expr.func == "cummax":
                    return pd.Series(
                        dfmacro_rs.cummax_parallel(col.to_numpy(dtype=float)),
                        index=df.index,
                    )
                elif expr.func == "cummin":
                    return pd.Series(
                        dfmacro_rs.cummin_parallel(col.to_numpy(dtype=float)),
                        index=df.index,
                    )
            except Exception:
                pass  # Fallback to Pandas

        # --- PANDAS FALLBACK ---
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
