"""DF wrapper: intercepts __getitem__ and adds query verbs."""

from __future__ import annotations

import sys
from typing import Any

from .compiler import compile_expr
from .expr import AggExpr, ColumnExpr, Expr


class GroupByProxy:
    def __init__(self, df: Any, by: Any) -> None:
        self._df = df
        self._by = by.name if hasattr(by, "name") else by

    def agg(self, **kwargs: Any) -> Any:
        if type(self._df).__module__.startswith("pandas"):
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
    __slots__ = ("_df",)

    def __init__(self, df: Any) -> None:
        object.__setattr__(self, "_df", df)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, Expr):
            mask = compile_expr(key, self._df)
            if type(self._df).__module__.startswith("pandas"):
                return DF(self._df[mask])
            else:
                return DF(self._df.filter(mask))
        if isinstance(key, str):
            return self._df[key]
        return DF(self._df[key]) if hasattr(self._df[key], "columns") else self._df[key]

    def select(self, *args: Any) -> Any:
        cols = [arg.name if hasattr(arg, "name") else arg for arg in args]
        if type(self._df).__module__.startswith("pandas"):
            return DF(self._df[cols])
        else:
            return DF(self._df.select(cols))

    def groupby(self, by: Any) -> GroupByProxy:
        return GroupByProxy(self._df, by)

    def mutate(self, **kwargs: Any) -> Any:
        if type(self._df).__module__.startswith("pandas"):
            new_df = self._df
            for col_name, expr in kwargs.items():
                mask = compile_expr(expr, self._df)
                new_df = new_df.assign(**{col_name: mask})
        else:
            exprs = []
            for col_name, expr in kwargs.items():
                compiled = compile_expr(expr, self._df)
                exprs.append(compiled.alias(col_name))
            new_df = self._df.with_columns(exprs)
        frame = sys._getframe(1)
        from .enable import _register_dynamic_name

        for col_name in kwargs.keys():
            if col_name not in frame.f_globals:
                frame.f_globals[col_name] = ColumnExpr(col_name)
                _register_dynamic_name(frame, col_name)
        return DF(new_df)

    def order_by(self, by: Any, descending: bool = False) -> Any:
        col_name = by.name if hasattr(by, "name") else by
        if type(self._df).__module__.startswith("pandas"):
            return DF(self._df.sort_values(by=col_name, ascending=not descending))
        else:
            return DF(self._df.sort(col_name, descending=descending))

    def join(self, other: Any, on: Any, how: str = "inner") -> Any:
        on_col = on.name if hasattr(on, "name") else on
        other_df = other.raw if isinstance(other, DF) else other
        if type(self._df).__module__.startswith("pandas"):
            return DF(self._df.merge(other_df, on=on_col, how=how))
        else:
            return DF(self._df.join(other_df, on=on_col, how=how))

    def head(self, n: int = 5) -> Any:
        return DF(self._df.head(n))

    def tail(self, n: int = 5) -> Any:
        return DF(self._df.tail(n))

    def slice(self, offset: int, length: int) -> Any:
        if type(self._df).__module__.startswith("pandas"):
            return DF(self._df.iloc[offset : offset + length])
        else:
            return DF(self._df.slice(offset, length))

    def drop(self, *args: Any) -> Any:
        cols = [arg.name if hasattr(arg, "name") else arg for arg in args]
        if type(self._df).__module__.startswith("pandas"):
            return DF(self._df.drop(columns=cols))
        else:
            return DF(self._df.drop(cols))

    def rename(self, **kwargs: Any) -> Any:
        mapping = {
            old: (new.name if hasattr(new, "name") else new)
            for old, new in kwargs.items()
        }
        if type(self._df).__module__.startswith("pandas"):
            return DF(self._df.rename(columns=mapping))
        else:
            return DF(self._df.rename(mapping))

    def drop_null(self, subset: Any = None) -> Any:
        subset_cols = [subset.name] if hasattr(subset, "name") else subset
        if type(self._df).__module__.startswith("pandas"):
            return DF(self._df.dropna(subset=subset_cols))
        else:
            return DF(self._df.drop_nulls(subset=subset_cols))

    def fill_null(self, col: Any, value: Any) -> Any:
        col_name = col.name if hasattr(col, "name") else col
        if type(self._df).__module__.startswith("pandas"):
            new_df = self._df.copy()
            new_df[col_name] = new_df[col_name].fillna(value)
            return DF(new_df)
        else:
            return DF(self._df.with_columns(pl.col(col_name).fill_null(value)))

    def sample(self, n: int = 1, seed: int | None = None) -> Any:
        if type(self._df).__module__.startswith("pandas"):
            return DF(self._df.sample(n=n, random_state=seed))
        else:
            return DF(self._df.sample(n=n, seed=seed))

    def explain(self) -> str:
        return repr(self._df)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._df, name)
        if callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                if hasattr(result, "columns"):
                    return DF(result)
                return result

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
        return self._df
