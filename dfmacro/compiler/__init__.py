"""Backend dispatch for dfmacro expression compilation."""

from __future__ import annotations

from typing import Any

from ..expr import Expr


class CompileError(TypeError):
    """Raised when an expression cannot be compiled for a backend."""


def _is_polars(df: Any) -> bool:
    try:
        import polars as pl
    except ImportError:
        return False
    return isinstance(df, pl.DataFrame)


def _is_pandas(df: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(df, pd.DataFrame)


def _is_dask_dataframe(df: Any) -> bool:
    try:
        import dask.dataframe as dd
    except ImportError:
        return False
    return isinstance(df, dd.DataFrame)


def compile_expr(expr: Expr, df: Any) -> Any:
    from ..plugins import backend_compiler_for

    plugin_compiler = backend_compiler_for(df)
    if plugin_compiler is not None:
        return plugin_compiler(expr, df)

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


def compile_to_sql(expr: Any, table_name: str = "df") -> str:
    """Compile an expression or LazyDF plan into DuckDB-compatible SQL.

    Args:
        expr: An ``Expr`` for a SQL expression fragment, or a ``LazyDF`` query
            plan for a complete ``SELECT`` statement.
        table_name: Existing SQL source table for a LazyDF plan. Defaults to
            ``"df"``; dotted schema and table names are supported.

    Returns:
        A SQL string. This function does not execute the generated query.

    Raises:
        CompileError: If the input or any planned operation is unsupported.
    """
    if isinstance(expr, Expr):
        from . import sql as _sql_backend

        return _sql_backend.compile_expr(expr)

    from ..lazy import LazyDF

    if isinstance(expr, LazyDF):
        from . import sql as _sql_backend

        columns = getattr(expr._df, "columns", None)
        return _sql_backend.compile_query(expr._plan, table_name, columns)

    raise CompileError(
        "compile_to_sql expects an Expr or LazyDF, "
        f"got {type(expr).__name__}."
    )