"""Dask backend using the pandas expression compiler per partition."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..expr import AggExpr, Expr
from . import CompileError
from .pandas import compile_expr as compile_pandas_expr


def _as_series(value: Any, frame: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return pd.Series(value, index=frame.index)
    return pd.Series(value, index=frame.index)


def compile_expr(expr: Expr, df: Any) -> Any:
    """Compile an expression to a lazy Dask Series by applying pandas per partition."""
    from dask.dataframe.utils import meta_nonempty

    sample = meta_nonempty(df._meta)
    sample_result = _as_series(compile_pandas_expr(expr, sample), sample)
    meta = sample_result.iloc[:0]

    def evaluate_partition(partition: pd.DataFrame) -> pd.Series:
        result = compile_pandas_expr(expr, partition)
        return _as_series(result, partition)

    return df.map_partitions(evaluate_partition, meta=meta)


def execute_join(left: Any, right: Any, on: Any, how: str = "inner") -> Any:
    """Execute a distributed Dask merge using Dask's native shuffle strategy."""
    import dask.dataframe as dd

    if not isinstance(left, dd.DataFrame) or not isinstance(right, dd.DataFrame):
        raise CompileError("Dask lazy joins require two Dask DataFrames.")
    keys = list(on) if isinstance(on, (list, tuple)) else [on]
    missing_left = set(keys) - set(left.columns)
    missing_right = set(keys) - set(right.columns)
    if missing_left or missing_right:
        raise CompileError(
            f"Join key(s) missing from Dask inputs: left={sorted(missing_left)!r}, "
            f"right={sorted(missing_right)!r}."
        )
    normalized_how = "outer" if how in {"full", "outer"} else how
    if normalized_how not in {"inner", "left", "right", "outer"}:
        raise CompileError(f"Unsupported Dask join type: {how!r}.")
    return dd.merge(left, right, on=keys, how=normalized_how)


def execute_groupby_agg(frame: Any, by: Any, aggregations: dict[str, Any]) -> Any:
    """Execute named aggregation expressions using Dask's distributed groupby."""
    keys = list(by) if isinstance(by, (list, tuple)) else [by]
    normalized: dict[str, list[str]] = {}
    output_names: dict[tuple[str, str], list[str]] = {}
    for output_name, expression in aggregations.items():
        if not isinstance(expression, AggExpr):
            raise CompileError(
                f"Unsupported Dask aggregation for {output_name!r}: "
                f"{type(expression).__name__}."
            )
        if expression.kwargs:
            raise CompileError(
                "Dask groupby aggregation does not currently support aggregation "
                f"arguments for {expression.func!r}."
            )
        column_name = expression.column.name
        if column_name not in frame.columns:
            raise CompileError(
                f"Aggregation column {column_name!r} not found in Dask DataFrame."
            )
        normalized.setdefault(column_name, []).append(expression.func)
        output_names.setdefault((column_name, expression.func), []).append(output_name)
    grouped = frame.groupby(keys).agg(normalized)
    output_columns = []
    for column in grouped.columns:
        if isinstance(column, tuple) and len(column) == 2:
            source_column, function = column
        else:
            source_column, function = column, ""
        names = output_names.get((source_column, function), [])
        if len(names) != 1:
            raise CompileError(
                "Dask groupby aggregation output names must be unique for each "
                f"{source_column!r}/{function!r} aggregation."
            )
        output_columns.append(names[0])
    grouped.columns = output_columns
    return grouped.reset_index()
