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


def _is_dask_dataframe(df: Any) -> bool:
    return type(df).__module__.startswith("dask.dataframe")


def compile_expr(expr: Expr, df: Any) -> Any:
    if not isinstance(expr, Expr):
        raise CompileError(
            f"compile_expr expects an Expr node, got {type(expr).__name__}."
        )

    if _is_polars(df):
        # Fix: Use exact type name check for LazyFrame instead of hasattr
        if type(df).__name__ == "LazyFrame":
            raise CompileError(
                "Polars LazyFrame support coming soon. "
                "Please call .collect() before using dfmacro."
            )
        from . import polars as _polars_backend
        return _polars_backend.compile_expr(expr, df)

    if _is_pandas(df):
        from . import pandas as _pandas_backend
        return _pandas_backend.compile_expr(expr, df)

    if _is_dask_dataframe(df):
        from . import dask as _dask_backend
        return _dask_backend.compile_expr(expr, df)

    raise CompileError(
        f"Unsupported DataFrame type: {type(df).__module__}."
    )


def compile_to_sql(expr: Expr) -> str:
    """Compile an expression tree into a SQL WHERE clause."""
    from . import sql as _sql_backend
    return _sql_backend.compile_expr(expr)