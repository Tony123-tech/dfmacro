"""Dask backend using the pandas expression compiler per partition."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..expr import Expr
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
