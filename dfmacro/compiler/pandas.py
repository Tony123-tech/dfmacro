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


def _arrow_table_for_column(df: Any, column_name: str) -> Any | None:
    """Create an optional Arrow view of one column for Rust acceleration.

    Args:
        df: Source Pandas DataFrame.
        column_name: Unique column name to export.

    Returns:
        A PyArrow table when PyArrow and conversion are available; otherwise
        ``None`` so the caller uses the Pandas implementation.
    """
    if not getattr(df.columns, "is_unique", False):
        return None
    try:
        import pyarrow as pa

        return pa.Table.from_pandas(df[[column_name]], preserve_index=False)
    except (ImportError, TypeError, ValueError):
        return None


def execute_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Any,
    how: str = "inner",
) -> pd.DataFrame:
    """Execute a validated lazy join with Pandas.

    Args:
        left: Left input DataFrame.
        right: Right input DataFrame.
        on: Shared join key or keys.
        how: Pandas merge strategy.

    Returns:
        The merged Pandas DataFrame.

    Raises:
        CompileError: If either input is not a Pandas DataFrame or keys are
            missing from an input.
        ValueError: If ``how`` is not supported.
    """
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise CompileError("Pandas lazy joins require two pandas DataFrames.")
    keys = list(on) if isinstance(on, (list, tuple)) else [on]
    if not keys or any(not isinstance(key, str) for key in keys):
        raise CompileError("Pandas lazy joins require one or more string column names.")
    missing_left = set(keys) - set(left.columns)
    missing_right = set(keys) - set(right.columns)
    if missing_left or missing_right:
        missing = []
        if missing_left:
            missing.append(f"left: {sorted(missing_left)!r}")
        if missing_right:
            missing.append(f"right: {sorted(missing_right)!r}")
        raise CompileError(f"Join key(s) missing from {'; '.join(missing)}.")
    normalized_how = "outer" if how == "full" else how
    if normalized_how not in {"inner", "left", "right", "outer"}:
        raise CompileError(f"Unsupported Pandas join type: {how!r}.")
    return left.merge(right, on=keys, how=normalized_how)


def compile_expr(expr: Expr, df: Any) -> Any:
    """Compile an expression tree to a Pandas value or Series.

    Args:
        expr: Expression node to compile.
        df: Pandas DataFrame used to resolve column references.

    Returns:
        The computed scalar, NumPy value, or Pandas Series.

    Raises:
        CompileError: If the expression is invalid or unsupported by Pandas.
    """
    if isinstance(expr, AccessorExpr):
        column = compile_expr(expr.column, df)
        if isinstance(expr, DatePartExpr) or expr.namespace == "dt":
            dtype = getattr(column, "dtype", None)
            if not pd.api.types.is_datetime64_any_dtype(dtype):
                column_name = (
                    expr.column.name
                    if isinstance(expr.column, ColumnExpr)
                    else repr(expr.column)
                )
                available = ", ".join(
                    f"{name}: {column_dtype}"
                    for name, column_dtype in df.dtypes.items()
                )
                raise CompileError(
                    f"Datetime accessor '.dt.{expr.method_name}' requires a "
                    f"datetime64 column; {column_name!r} has dtype {dtype!r}. "
                    f"Available columns and dtypes: {available}."
                )
        elif isinstance(expr, StringMethodExpr) or expr.namespace == "str":
            dtype = getattr(column, "dtype", None)
            if not (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
            ):
                column_name = (
                    expr.column.name
                    if isinstance(expr.column, ColumnExpr)
                    else repr(expr.column)
                )
                available = ", ".join(
                    f"{name}: {column_dtype}"
                    for name, column_dtype in df.dtypes.items()
                )
                raise CompileError(
                    f"String accessor '.str.{expr.method_name}' requires an "
                    f"object or string column; {column_name!r} has dtype "
                    f"{dtype!r}. Available columns and dtypes: {available}."
                )
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
                    table = _arrow_table_for_column(df, expr.left.name)
                    if table is not None and hasattr(dfmacro_rs, "filter_compare_arrow"):
                        mask = dfmacro_rs.filter_compare_arrow(
                            table, expr.left.name, expr.op, scalar=float(right)
                        )
                        return pd.Series(mask, index=df.index)

                # Case 2: Literal OP Column
                elif isinstance(expr.right, ColumnExpr) and isinstance(
                    expr.left, Literal
                ):
                    table = _arrow_table_for_column(df, expr.right.name)
                    if table is not None and hasattr(dfmacro_rs, "filter_compare_arrow"):
                        mask = dfmacro_rs.filter_compare_arrow(
                            table,
                            expr.right.name,
                            _FLIP_OP[expr.op],
                            scalar=float(left),
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
                if expr.func in {"var", "std", "cumsum", "cummax", "cummin"} and isinstance(
                    expr.column, ColumnExpr
                ):
                    table = _arrow_table_for_column(df, expr.column.name)
                    if table is not None and hasattr(dfmacro_rs, "numeric_window_arrow"):
                        result = dfmacro_rs.numeric_window_arrow(
                            table, expr.column.name, expr.func
                        )
                        if expr.func in {"var", "std"}:
                            return result
                        return pd.Series(result, index=df.index)
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
