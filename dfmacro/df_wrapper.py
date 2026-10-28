"""DataFrame wrapper exposing a backend-neutral collection of query verbs.

The wrapper keeps DataFrame execution eager while delegating expression
compilation to the active backend. Use :meth:`DF.lazy` to record a plan first.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from collections.abc import Callable, Iterable
from typing import Any

from .compiler import _is_dask_dataframe, compile_expr
from .expr import AggExpr, Expr, Literal
from .optimizer import optimize_expr, optimize_plan
from .resilience import retry


def _is_pandas(obj: Any) -> bool:
    return type(obj).__module__.startswith("pandas")


def _raise_if_dask_unsupported(df: Any, operation: str) -> None:
    if _is_dask_dataframe(df):
        raise NotImplementedError(
            f"{operation} is not currently supported for Dask DataFrames."
        )


def _resolve_column_name(col: Any) -> Any:
    """Resolve an Expr / Series / string to a column name (string)."""
    if hasattr(col, "name") and isinstance(col.name, str):
        return col.name
    return col


def _resolve_column_names(columns: Any) -> Any:
    if columns is None:
        return None
    if isinstance(columns, (list, tuple)):
        return [_resolve_column_name(column) for column in columns]
    return _resolve_column_name(columns)


def _matrix_to_polars(values: Any, columns: list[str]) -> Any:
    import polars as pl

    index_name = "variable"
    while index_name in columns:
        index_name = f"_{index_name}"
    data = {index_name: columns}
    data.update({column: values[:, i] for i, column in enumerate(columns)})
    return pl.DataFrame(data)


def _unwrap_dataframe(frame: Any) -> Any:
    return frame.raw if isinstance(frame, DF) else frame


def _expression_cache_key(expr: Expr) -> str | None:
    """Return a stable cache key, or None when an expression cannot be serialized."""
    try:
        serialized = pickle.dumps(expr, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # Arbitrary literal values can define unpickleable state; skip caching them.
        return None
    return hashlib.sha256(serialized).hexdigest()


def _wrap_dataframe_result(result: Any) -> Any:
    if isinstance(result, DF):
        return result
    return DF(result) if hasattr(result, "columns") else result


def _get_operation(target: Any, name: str, context: str) -> Callable[..., Any]:
    try:
        operation = getattr(target, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported {context} function {name!r}.") from exc
    if not callable(operation):
        raise ValueError(f"Unsupported {context} function {name!r}.")
    return operation


class GroupByProxy:
    """Backend-neutral grouping proxy used by :meth:`DF.groupby`.

    Args:
        df: The wrapped backend DataFrame.
        by: One grouping column or a sequence of grouping columns.
    """

    _df: Any
    _by: Any

    def __init__(self, df: Any, by: Any) -> None:
        _raise_if_dask_unsupported(df, "Groupby aggregation")
        self._df = df
        self._by = _resolve_column_name(by)

    def agg(self, **kwargs: Any) -> DF:
        """Aggregate expression columns and return a wrapped DataFrame.

        Args:
            **kwargs: Output names mapped to aggregation expressions.

        Returns:
            A ``DF`` containing one row per group.
        """
        if _is_pandas(self._df):
            pd_aggs = {}
            for out_col, expr in kwargs.items():
                if isinstance(expr, AggExpr):
                    if expr.kwargs:
                        pd_aggs[out_col] = (
                            expr.column.name,
                            lambda x, f=expr.func, kw=expr.kwargs: getattr(x, f)(**kw),
                        )
                    else:
                        pd_aggs[out_col] = (expr.column.name, expr.func)
                else:
                    raise TypeError(f"Unsupported agg expression: {expr}")
            return DF(self._df.groupby(self._by).agg(**pd_aggs))
        else:
            import polars as pl

            aggs = []
            for out_col, expr in kwargs.items():
                if isinstance(expr, AggExpr):
                    if expr.kwargs:
                        col_expr = getattr(pl.col(expr.column.name), expr.func)(
                            **expr.kwargs
                        )
                    else:
                        col_expr = getattr(pl.col(expr.column.name), expr.func)()
                    aggs.append(col_expr.alias(out_col))
                else:
                    raise TypeError(f"Unsupported agg expression: {expr}")
            return DF(self._df.group_by(self._by).agg(aggs))


class DF:
    """Wrap a Pandas, Polars, or supported Dask DataFrame.

    Expressions are compiled for the underlying backend when indexed with an
    :class:`~dfmacro.expr.Expr`. Results that are DataFrames remain wrapped;
    access the native object through :attr:`raw`.

    The wrapper exposes filtering, selection, mutation, sorting, grouping,
    aggregation, reshaping, joins, null handling, and numeric transforms.
    Backend-specific limitations are reported with ``NotImplementedError``.

    Args:
        df: A supported DataFrame instance.
    """

    __slots__ = ("_cache", "_df")
    _df: Any
    _cache: dict[str, DF]

    def __init__(self, df: Any) -> None:
        """Initialize a wrapper and its per-instance expression cache.

        Args:
            df: The native DataFrame to wrap.
        """
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_cache", {})

    def _transform_column(
        self,
        col: Any,
        pandas_operation: Callable[[Any], Any],
        polars_operation: Callable[[Any], Any],
    ) -> DF:
        col_name = _resolve_column_name(col)
        if _is_pandas(self._df):
            result = pandas_operation(self._df[col_name])
            return DF(result.to_frame(name=col_name))

        import polars as pl

        expression = polars_operation(pl.col(col_name)).alias(col_name)
        return DF(self._df.select(expression))

    def _transform_numeric_columns(
        self,
        pandas_operation: Callable[[Any], Any],
        polars_operation: Callable[[Any], Any],
    ) -> DF:
        if _is_pandas(self._df):
            numeric_columns = self._df.select_dtypes(include="number").columns
            result = self._df.copy()
            result[numeric_columns] = pandas_operation(self._df[numeric_columns])
            return DF(result)

        import polars as pl

        return DF(self._df.with_columns(polars_operation(pl.selectors.numeric())))

    def _membership_join(self, other: Any, on: Any, matched: bool) -> DF:
        _raise_if_dask_unsupported(self._df, "Semi/anti joins")
        on_cols = _resolve_column_names(on)
        other_df = _unwrap_dataframe(other)
        if not _is_pandas(self._df):
            how = "semi" if matched else "anti"
            return DF(self._df.join(other_df, on=on_cols, how=how))

        marker = "__dfmacro_merge_marker__"
        while marker in self._df.columns or marker in other_df.columns:
            marker = f"_{marker}"
        key_columns = on_cols if isinstance(on_cols, list) else [on_cols]
        keys = other_df[key_columns].drop_duplicates()
        merged = self._df.merge(
            keys, on=on_cols, how="left", indicator=marker, sort=False
        )
        match_value = "both" if matched else "left_only"
        return DF(merged.loc[merged[marker] == match_value, self._df.columns])

    def _conditional_replace(self, cond: Any, other: Any, keep_matching: bool) -> DF:
        if isinstance(cond, Expr):
            cond = compile_expr(cond, self._df)
        if _is_pandas(self._df):
            method = self._df.where if keep_matching else self._df.mask
            return DF(method(cond, other))

        import polars as pl

        replacement = None if other is None else pl.lit(other)
        if keep_matching:
            expressions = [
                pl.when(cond).then(pl.col(column)).otherwise(replacement).alias(column)
                for column in self._df.columns
            ]
        else:
            expressions = [
                pl.when(cond).then(replacement).otherwise(pl.col(column)).alias(column)
                for column in self._df.columns
            ]
        return DF(self._df.with_columns(expressions))

    def __getitem__(self, key: Any) -> Any:
        """Filter with an expression or delegate ordinary indexing.

        Args:
            key: An expression predicate, column name, or backend indexer.

        Returns:
            A wrapped filtered DataFrame, a native selected column, or a
            wrapped backend DataFrame result.
        """
        if isinstance(key, Expr):
            key = optimize_expr(key)
            cache_key = _expression_cache_key(key)
            if cache_key is not None and cache_key in self._cache:
                return self._cache[cache_key]

            if isinstance(key, Literal) and type(key.value) is bool:
                if key.value:
                    selected = self._df.copy() if _is_pandas(self._df) else self._df
                elif _is_pandas(self._df):
                    selected = self._df.iloc[0:0]
                else:
                    selected = self._df.head(0)
                result = DF(selected)
                if cache_key is not None:
                    self._cache[cache_key] = result
                return result

            mask = compile_expr(key, self._df)
            if _is_pandas(self._df):
                result = DF(self._df[mask])
            elif _is_dask_dataframe(self._df):
                result = DF(self._df[mask])
            else:
                result = DF(self._df.filter(mask))

            if cache_key is not None:
                self._cache[cache_key] = result
            return result
        if isinstance(key, str):
            return self._df[key]
        result = self._df[key]
        return _wrap_dataframe_result(result)

    def select(self, *args: Any) -> DF:
        """Select the named columns and return a new wrapper.

        Args:
            *args: Column names or column expressions.

        Returns:
            A ``DF`` containing the requested columns.
        """
        cols = [arg.name if hasattr(arg, "name") else arg for arg in args]
        if _is_pandas(self._df):
            return DF(self._df[cols])
        if _is_dask_dataframe(self._df):
            return DF(self._df[cols])
        else:
            return DF(self._df.select(cols))

    def groupby(self, by: Any) -> GroupByProxy:
        """Create a grouping proxy.

        Args:
            by: Grouping column or columns.

        Returns:
            A proxy whose ``agg`` method accepts named aggregation expressions.
        """
        return GroupByProxy(self._df, by)

    def mutate(self, **kwargs: Any) -> DF:
        """Add or replace columns using expressions or literal values.

        Args:
            **kwargs: Output column names mapped to expressions or values.

        Returns:
            A new ``DF`` with the requested columns.
        """
        if _is_pandas(self._df) or _is_dask_dataframe(self._df):
            new_df = self._df
            for col_name, expr in kwargs.items():
                if isinstance(expr, Expr):
                    value = compile_expr(expr, self._df)
                else:
                    value = expr
                new_df = new_df.assign(**{col_name: value})
        else:
            import polars as pl

            exprs = []
            for col_name, expr in kwargs.items():
                if isinstance(expr, Expr):
                    compiled = compile_expr(expr, self._df)
                    exprs.append(compiled.alias(col_name))
                else:
                    if isinstance(expr, pl.Expr):
                        exprs.append(expr.alias(col_name))
                    else:
                        exprs.append(pl.lit(expr).alias(col_name))
            new_df = self._df.with_columns(exprs)
        return DF(new_df)

    def where(self, cond: Any, other: Any = None) -> DF:
        return self._conditional_replace(cond, other, keep_matching=True)

    def mask(self, cond: Any, other: Any = None) -> DF:
        return self._conditional_replace(cond, other, keep_matching=False)

    def nlargest(self, n: int, col: Any) -> DF:
        col_name = _resolve_column_name(col)
        if _is_pandas(self._df):
            return DF(self._df.nlargest(n, col_name))
        else:
            return DF(self._df.sort(col_name, descending=True).head(n))

    def nsmallest(self, n: int, col: Any) -> DF:
        col_name = _resolve_column_name(col)
        if _is_pandas(self._df):
            return DF(self._df.nsmallest(n, col_name))
        else:
            return DF(self._df.sort(col_name).head(n))

    def order_by(self, by: Any, descending: bool = False) -> DF:
        """Sort rows by one column.

        Args:
            by: Column name or expression.
            descending: Whether to sort in descending order.

        Returns:
            A sorted ``DF``.
        """
        col_name = _resolve_column_name(by)
        if _is_pandas(self._df):
            return DF(self._df.sort_values(by=col_name, ascending=not descending))
        if _is_dask_dataframe(self._df):
            return DF(self._df.sort_values(by=col_name, ascending=not descending))
        else:
            return DF(self._df.sort(col_name, descending=descending))

    def sort_values(self, by: Any, ascending: Any = True) -> DF:
        by = (
            [_resolve_column_name(col) for col in by]
            if isinstance(by, (list, tuple))
            else _resolve_column_name(by)
        )
        if _is_pandas(self._df):
            return DF(self._df.sort_values(by=by, ascending=ascending))
        if _is_dask_dataframe(self._df):
            return DF(self._df.sort_values(by=by, ascending=ascending))
        descending = (
            [not value for value in ascending]
            if isinstance(ascending, (list, tuple))
            else not ascending
        )
        return DF(self._df.sort(by, descending=descending))

    def reset_index(self, *args: Any, **kwargs: Any) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.reset_index(*args, **kwargs))
        unsupported = kwargs.keys() - {"drop"}
        if args or unsupported:
            raise NotImplementedError(
                "reset_index options are not supported for Polars DataFrames."
            )
        return DF(self._df.clone())

    def set_index(self, keys: Any, drop: bool = True) -> DF:
        if not _is_pandas(self._df):
            raise NotImplementedError(
                "Polars DataFrames do not support pandas-style indexes."
            )
        keys = (
            [_resolve_column_name(key) for key in keys]
            if isinstance(keys, (list, tuple))
            else _resolve_column_name(keys)
        )
        return DF(self._df.set_index(keys, drop=drop))

    def assign(self, **kwargs: Any) -> DF:
        return self.mutate(**kwargs)

    def filter(
        self,
        items: Any = None,
        like: str | None = None,
        regex: str | None = None,
    ) -> DF:
        if _is_pandas(self._df):
            if isinstance(items, Expr):
                items = compile_expr(items, self._df)
            if hasattr(items, "dtype"):
                import pandas as pd

                if pd.api.types.is_bool_dtype(items.dtype):
                    return DF(self._df.loc[items])
            return DF(self._df.filter(items=items, like=like, regex=regex))

        import polars as pl

        if isinstance(items, Expr):
            items = compile_expr(items, self._df)
        if isinstance(items, pl.Expr):
            return DF(self._df.filter(items))
        if isinstance(items, pl.Series) and items.dtype == pl.Boolean:
            return DF(self._df.filter(items))
        if items is not None or like is not None or regex is not None:
            raise TypeError("Polars filter expects a boolean expression.")
        raise TypeError("filter expects a boolean expression or pandas label filters.")

    def isin(self, values: Any) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.isin(values))

        import polars as pl

        if isinstance(values, dict):
            expressions = [
                pl.col(column).is_in(values.get(column, [])).alias(column)
                for column in self._df.columns
            ]
        else:
            expressions = [
                pl.col(column).is_in(values).alias(column)
                for column in self._df.columns
            ]
        return DF(self._df.select(expressions))

    def between(
        self,
        col: Any,
        lower: Any,
        upper: Any,
        inclusive: str = "both",
    ) -> DF:
        col_name = _resolve_column_name(col)
        if _is_pandas(self._df):
            mask = self._df[col_name].between(lower, upper, inclusive=inclusive)
            return DF(self._df[mask])

        import polars as pl

        closed = {
            "both": "both",
            "neither": "none",
            "left": "left",
            "right": "right",
        }.get(inclusive)
        if closed is None:
            raise ValueError(
                "inclusive must be one of 'both', 'neither', 'left', or 'right'"
            )
        return DF(
            self._df.filter(pl.col(col_name).is_between(lower, upper, closed=closed))
        )

    def describe(self) -> DF:
        return DF(self._df.describe())

    def profile(self) -> dict:
        """Profile the DataFrame to extract schema and statistical information.

        Returns:
            Dictionary containing profiling information with columns, dtypes,
            null counts, unique counts, and numeric/categorical statistics.
        """
        from .profiling import profile as profile_df

        return profile_df(self._df)

    def value_counts(self, col: Any) -> DF:
        col_name = _resolve_column_name(col)
        if _is_pandas(self._df):
            counts = self._df[col_name].value_counts()
            return DF(counts.rename_axis(col_name).reset_index(name="count"))
        return DF(self._df[col_name].value_counts(sort=True))

    def corr(self) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.corr(numeric_only=True))

        import polars as pl

        numeric_df = self._df.select(pl.selectors.numeric())
        columns = numeric_df.columns
        if not columns:
            return DF(pl.DataFrame())
        values = numeric_df.corr()
        return DF(_matrix_to_polars(values, columns))

    def cov(self) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.cov(numeric_only=True))

        import numpy as np
        import polars as pl

        numeric_df = self._df.select(pl.selectors.numeric())
        columns = numeric_df.columns
        if not columns:
            return DF(pl.DataFrame())
        values = np.atleast_2d(np.cov(numeric_df.to_numpy(), rowvar=False))
        return DF(_matrix_to_polars(values, columns))

    def pct_change(self, col: Any = None, periods: int = 1) -> DF:
        if col is None:
            if _is_pandas(self._df):
                return DF(self._df.pct_change(periods=periods))
            import polars as pl

            return DF(self._df.select(pl.all().pct_change(periods)))
        return self._transform_column(
            col,
            lambda series: series.pct_change(periods=periods),
            lambda expression: expression.pct_change(periods),
        )

    def shift(self, col: Any = None, periods: int = 1) -> DF:
        if col is None:
            if _is_pandas(self._df):
                return DF(self._df.shift(periods=periods))
            import polars as pl

            return DF(self._df.select(pl.all().shift(periods)))
        col_name = _resolve_column_name(col)
        if col_name not in self._df.columns and isinstance(col_name, int):
            return self.shift(periods=col_name)
        return self._transform_column(
            col,
            lambda series: series.shift(periods=periods),
            lambda expression: expression.shift(periods),
        )

    def diff(self, col: Any = None, periods: int = 1) -> DF:
        if col is None:
            if _is_pandas(self._df):
                return DF(self._df.diff(periods=periods))
            import polars as pl

            return DF(self._df.select(pl.all().diff(periods)))
        col_name = _resolve_column_name(col)
        if col_name not in self._df.columns and isinstance(col_name, int):
            return self.diff(periods=col_name)
        return self._transform_column(
            col,
            lambda series: series.diff(periods=periods),
            lambda expression: expression.diff(periods),
        )

    def cumsum(self, col: Any = None) -> DF:
        if col is None:
            if _is_pandas(self._df):
                return DF(self._df.cumsum())
            import polars as pl

            return DF(self._df.select(pl.all().cum_sum()))
        return self._transform_column(
            col, lambda series: series.cumsum(), lambda expression: expression.cum_sum()
        )

    def cummax(self, col: Any = None) -> DF:
        if col is None:
            if _is_pandas(self._df):
                return DF(self._df.cummax())
            import polars as pl

            return DF(self._df.select(pl.all().cum_max()))
        return self._transform_column(
            col, lambda series: series.cummax(), lambda expression: expression.cum_max()
        )

    def cummin(self, col: Any = None) -> DF:
        if col is None:
            if _is_pandas(self._df):
                return DF(self._df.cummin())
            import polars as pl

            return DF(self._df.select(pl.all().cum_min()))
        return self._transform_column(
            col, lambda series: series.cummin(), lambda expression: expression.cum_min()
        )

    def melt(self, id_vars: Any = None, value_vars: Any = None) -> DF:
        id_vars = _resolve_column_names(id_vars)
        value_vars = _resolve_column_names(value_vars)
        if _is_pandas(self._df):
            return DF(self._df.melt(id_vars=id_vars, value_vars=value_vars))
        return DF(self._df.unpivot(on=value_vars, index=id_vars))

    def pivot(self, index: Any, columns: Any, values: Any) -> DF:
        index = _resolve_column_names(index)
        columns = _resolve_column_names(columns)
        values = _resolve_column_names(values)
        if _is_pandas(self._df):
            return DF(self._df.pivot(index=index, columns=columns, values=values))
        return DF(self._df.pivot(on=columns, index=index, values=values))

    def pivot_table(
        self, index: Any, columns: Any, values: Any, aggfunc: Any = "mean"
    ) -> DF:
        index = _resolve_column_names(index)
        columns = _resolve_column_names(columns)
        values = _resolve_column_names(values)
        if _is_pandas(self._df):
            return DF(
                self._df.pivot_table(
                    index=index, columns=columns, values=values, aggfunc=aggfunc
                )
            )
        return DF(
            self._df.pivot(
                on=columns,
                index=index,
                values=values,
                aggregate_function=aggfunc,
            )
        )

    def rolling(self, window: Any, col: Any, func: str = "mean") -> DF:
        return self._transform_column(
            col,
            lambda series: _get_operation(series.rolling(window), func, "rolling")(),
            lambda expression: _get_operation(expression, f"rolling_{func}", "rolling")(
                window_size=window
            ),
        )

    def ewm(self, span: float, col: Any, func: str = "mean") -> DF:
        return self._transform_column(
            col,
            lambda series: _get_operation(series.ewm(span=span), func, "EWM")(),
            lambda expression: _get_operation(expression, f"ewm_{func}", "EWM")(
                span=span
            ),
        )

    def resample(self, rule: str, col: Any, func: str = "sum", on: Any = None) -> DF:
        col_name = _resolve_column_name(col)
        if _is_pandas(self._df):
            result = _get_operation(
                self._df[col_name].resample(rule), func, "resample"
            )()
            return DF(result.to_frame(name=col_name))

        import polars as pl

        if on is None:
            datetime_columns = [
                name
                for name, dtype in self._df.schema.items()
                if dtype == pl.Date or isinstance(dtype, pl.Datetime)
            ]
            if len(datetime_columns) != 1:
                raise ValueError(
                    "Polars resample requires exactly one Date/Datetime column; "
                    "pass it with on= when the frame has zero or multiple."
                )
            on = datetime_columns[0]
        on_name = _resolve_column_name(on)
        aggregation = _get_operation(pl.col(col_name), func, "resample")()
        return DF(
            self._df.sort(on_name)
            .group_by_dynamic(index_column=on_name, every=rule, start_by="datapoint")
            .agg(aggregation)
        )

    def asfreq(self, freq: str) -> DF:
        if not _is_pandas(self._df):
            raise NotImplementedError(
                "asfreq is only supported for pandas DataFrames with a DatetimeIndex."
            )
        return DF(self._df.asfreq(freq))

    def interpolate(self, method: str = "linear") -> DF:
        if _is_pandas(self._df):
            return DF(self._df.interpolate(method=method))
        if method != "linear":
            raise NotImplementedError(
                "Polars DataFrames support only linear interpolation."
            )
        import polars as pl

        expressions = [
            pl.col(name).interpolate() if dtype.is_numeric() else pl.col(name)
            for name, dtype in self._df.schema.items()
        ]
        return DF(self._df.with_columns(expressions))

    def ffill(self) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.ffill())
        return DF(self._df.fill_null(strategy="forward"))

    def bfill(self) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.bfill())
        return DF(self._df.fill_null(strategy="backward"))

    def nunique(self) -> Any:
        if _is_pandas(self._df):
            return self._df.nunique()
        import polars as pl

        return DF(self._df.select(pl.all().n_unique()))

    def mode(self) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.mode())
        import polars as pl

        return DF(self._df.select(pl.all().mode()))

    def skew(self) -> Any:
        if _is_pandas(self._df):
            return self._df.skew(numeric_only=True)
        import polars as pl

        return DF(self._df.select(pl.selectors.numeric().skew(bias=False)))

    def kurt(self) -> Any:
        if _is_pandas(self._df):
            return self._df.kurt(numeric_only=True)
        import polars as pl

        return DF(
            self._df.select(pl.selectors.numeric().kurtosis(fisher=True, bias=False))
        )

    def merge(self, other: Any, on: Any, how: str = "inner") -> DF:
        _raise_if_dask_unsupported(self._df, "Merge")
        on_cols = _resolve_column_names(on)
        other_df = _unwrap_dataframe(other)
        if _is_pandas(self._df):
            return DF(self._df.merge(other_df, on=on_cols, how=how))
        return DF(self._df.join(other_df, on=on_cols, how=how))

    @staticmethod
    def concat(frames: Iterable[Any], axis: int = 0) -> DF:
        frames = [_unwrap_dataframe(frame) for frame in frames]
        if not frames:
            raise ValueError("concat requires at least one DataFrame.")
        dask_frame = next(
            (frame for frame in frames if _is_dask_dataframe(frame)), None
        )
        if dask_frame is not None:
            _raise_if_dask_unsupported(dask_frame, "Concatenation")
        if _is_pandas(frames[0]):
            import pandas as pd

            return DF(pd.concat(frames, axis=axis))
        import polars as pl

        if axis == 0:
            return DF(pl.concat(frames, how="diagonal"))
        if axis == 1:
            return DF(pl.concat(frames, how="horizontal"))
        raise ValueError("axis must be 0 or 1.")

    def merge_asof(self, other: Any, on: Any) -> DF:
        _raise_if_dask_unsupported(self._df, "As-of merge")
        on_col = _resolve_column_name(on)
        other_df = _unwrap_dataframe(other)
        if _is_pandas(self._df):
            import pandas as pd

            return DF(pd.merge_asof(self._df, other_df, on=on_col))
        return DF(self._df.sort(on_col).join_asof(other_df.sort(on_col), on=on_col))

    def cross_join(self, other: Any) -> DF:
        _raise_if_dask_unsupported(self._df, "Cross join")
        other_df = _unwrap_dataframe(other)
        if _is_pandas(self._df):
            return DF(self._df.merge(other_df, how="cross"))
        return DF(self._df.join(other_df, how="cross"))

    def semi_join(self, other: Any, on: Any) -> DF:
        return self._membership_join(other, on, matched=True)

    def anti_join(self, other: Any, on: Any) -> DF:
        return self._membership_join(other, on, matched=False)

    def apply(self, func: Callable[..., Any], axis: int = 0) -> Any:
        if not _is_pandas(self._df):
            raise NotImplementedError("apply is only supported for pandas DataFrames.")
        result = self._df.apply(func, axis=axis)
        return _wrap_dataframe_result(result)

    def pipe(self, func: Callable[[Any], Any]) -> Any:
        return _wrap_dataframe_result(func(self._df))

    def clip(self, lower: Any = None, upper: Any = None) -> DF:
        return self._transform_numeric_columns(
            lambda frame: frame.clip(lower=lower, upper=upper),
            lambda columns: columns.clip(lower_bound=lower, upper_bound=upper),
        )

    def abs(self) -> DF:
        return self._transform_numeric_columns(
            lambda frame: frame.abs(), lambda columns: columns.abs()
        )

    def round(self, decimals: int = 0) -> DF:
        return self._transform_numeric_columns(
            lambda frame: frame.round(decimals),
            lambda columns: columns.round(decimals=decimals),
        )

    def log(self, base: float = math.e) -> DF:
        if base <= 0 or base == 1 or not math.isfinite(base):
            raise ValueError("base must be a finite positive number other than 1.")
        import numpy as np

        return self._transform_numeric_columns(
            lambda frame: np.log(frame) / math.log(base),
            lambda columns: columns.log(base=base),
        )

    def sqrt(self) -> DF:
        import numpy as np

        return self._transform_numeric_columns(
            lambda frame: np.sqrt(frame),
            lambda columns: columns.sqrt(),
        )

    def pow(self, exponent: Any) -> DF:
        return self._transform_numeric_columns(
            lambda frame: frame.pow(exponent),
            lambda columns: columns.pow(exponent),
        )

    def join(self, other: Any, on: Any, how: str = "inner") -> DF:
        _raise_if_dask_unsupported(self._df, "Join")
        on_col = _resolve_column_name(on)
        other_df = _unwrap_dataframe(other)
        if _is_pandas(self._df):
            return DF(self._df.merge(other_df, on=on_col, how=how))
        else:
            return DF(self._df.join(other_df, on=on_col, how=how))

    def head(self, n: int = 5) -> DF:
        """Return the first ``n`` rows.

        Args:
            n: Number of rows to return, following backend ``head`` semantics.

        Returns:
            A ``DF`` containing the requested leading rows.
        """
        # Route through the same plan optimizer used by LazyDF pipelines.
        optimized_plan = optimize_plan([("head", (n,), {})])
        _, args, _ = optimized_plan[0]
        n = args[0]
        return DF(self._df.head(n))

    def tail(self, n: int = 5) -> DF:
        """Return the last ``n`` rows.

        Args:
            n: Number of rows to return, following backend ``tail`` semantics.

        Returns:
            A ``DF`` containing the requested trailing rows.
        """
        return DF(self._df.tail(n))

    def slice(self, offset: int, length: int) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.iloc[offset : offset + length])
        else:
            return DF(self._df.slice(offset, length))

    def drop(self, *args: Any) -> DF:
        cols = [_resolve_column_name(arg) for arg in args]
        if _is_pandas(self._df):
            return DF(self._df.drop(columns=cols))
        else:
            return DF(self._df.drop(cols))

    def rename(self, **kwargs: Any) -> DF:
        mapping = {
            old: (new.name if hasattr(new, "name") else new)
            for old, new in kwargs.items()
        }
        if _is_pandas(self._df):
            return DF(self._df.rename(columns=mapping))
        else:
            return DF(self._df.rename(mapping))

    def drop_null(self, subset: Any = None) -> DF:
        subset_cols = [_resolve_column_name(subset)] if subset is not None else None
        if _is_pandas(self._df):
            return DF(self._df.dropna(subset=subset_cols))
        else:
            return DF(self._df.drop_nulls(subset=subset_cols))

    def fill_null(self, col: Any, value: Any) -> DF:
        col_name = _resolve_column_name(col)
        if _is_pandas(self._df):
            new_df = self._df.copy()
            new_df[col_name] = new_df[col_name].fillna(value)
            return DF(new_df)
        else:
            import polars as pl

            return DF(self._df.with_columns(pl.col(col_name).fill_null(value)))

    def sample(self, n: int = 1, seed: int | None = None) -> DF:
        if _is_pandas(self._df):
            return DF(self._df.sample(n=n, random_state=seed))
        else:
            return DF(self._df.sample(n=n, seed=seed))

    def explain(self) -> str:
        """Return a readable summary of the eager DataFrame and its columns."""
        columns = getattr(self._df, "columns", ())
        if _is_pandas(self._df):
            details = [f"{name}: {dtype}" for name, dtype in self._df.dtypes.items()]
        else:
            schema = getattr(self._df, "schema", {})
            try:
                details = [f"{name}: {dtype}" for name, dtype in schema.items()]
            except AttributeError:
                details = [str(name) for name in columns]
        lines = [f"DataFrame ({type(self._df).__module__}.{type(self._df).__name__})"]
        lines.extend(f"  {item}" for item in details)
        return "\n".join(lines)

    def generate_pytest_code(
        self, filename: str, test_name: str = "test_dataframe_schema"
    ) -> str:
        """Generate pytest code to validate DataFrame schema and constraints.

        Args:
            filename: Name of the test file to generate
            test_name: Name of the test function

        Returns:
            String containing pytest code
        """
        from .profiling import generate_pytest_code as generate_pytest

        return generate_pytest(self._df, filename, test_name)

    @retry()
    def collect(self) -> DF:
        """Materialize a native lazy frame, or return this eager wrapper.

        Returns:
            A wrapper around the materialized backend DataFrame.
        """
        collect_method = getattr(self._df, "collect", None)
        if not callable(collect_method):
            return self
        result = collect_method()
        return _wrap_dataframe_result(result)

    def lazy(self) -> Any:
        """Return a lazy plan builder over this DataFrame.

        Returns:
            A :class:`dfmacro.lazy.LazyDF` that records supported operations
            until ``collect`` is called.
        """
        from .lazy import LazyDF

        return LazyDF(self._df)

    def to_duckdb(self) -> Any:
        """Create a DuckDB-backed wrapper for SQL execution.

        Returns:
            A ``DuckDBDF`` registered over the current DataFrame.
        """
        from .duckdb_wrapper import DuckDBDF

        return DuckDBDF(self._df)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._df, name)
        if callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                return _wrap_dataframe_result(result)

            return wrapper
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._df, name, value)

    def __repr__(self) -> str:
        return f"DF({self._df!r})"

    def __len__(self) -> int:
        return len(self._df)

    @property
    def raw(self) -> Any:
        """Return the native backend DataFrame."""
        return self._df
