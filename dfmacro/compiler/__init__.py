"""Backend dispatch for dfmacro expression compilation."""

from __future__ import annotations

from typing import Any

from ..expr import Expr


class CompileError(TypeError):
    """Raised when an expression cannot be compiled for a backend."""


def _is_polars(df: Any) -> bool:
    return type(df).__module__.startswith("polars")


def _is_pandas(df: Any) -> bool:
    return type(df).__module__.startswith("pandas")


def compile_expr(expr: Expr, df: Any) -> Any:
    if not isinstance(expr, Expr):
        raise CompileError(
            f"compile_expr expects an Expr node, got {type(expr).__name__}."
        )

    if _is_polars(df):
        from . import polars as _polars_backend

        # Check if it's a LazyFrame
        if hasattr(df, "collect") and not hasattr(df, "filter"):
            raise CompileError("Polars LazyFrame support coming soon.")
        return _polars_backend.compile_expr(expr, df)

    if _is_pandas(df):
        from . import pandas as _pandas_backend

        return _pandas_backend.compile_expr(expr, df)

    raise CompileError(f"Unsupported DataFrame type: {type(df).__module__}.")


def compile_to_sql(expr: Expr) -> str:
    """Compile an expression tree into a SQL WHERE clause."""
    from . import sql as _sql_backend

    return _sql_backend.compile_expr(expr)
