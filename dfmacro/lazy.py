"""Lazy evaluation engine for dfmacro."""

from __future__ import annotations

from typing import Any

from .expr import Expr


class LazyDF:
    """A lazy wrapper that records operations to be executed on .collect()."""

    __slots__ = ("_df", "_plan")

    def __init__(
        self, df: Any, plan: list[tuple[str, tuple, dict]] | None = None
    ) -> None:
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_plan", plan or [])

    def __getitem__(self, key: Any) -> LazyDF:
        if isinstance(key, Expr):
            return LazyDF(self._df, self._plan + [("filter", (key,), {})])
        raise TypeError(
            "LazyDF only supports expression filtering (e.g. df[price > 100]) for now."
        )

    def mutate(self, **kwargs: Any) -> LazyDF:
        return LazyDF(self._df, self._plan + [("mutate", (), kwargs)])

    def select(self, *args: Any) -> LazyDF:
        return LazyDF(self._df, self._plan + [("select", args, {})])

    def groupby(self, by: Any) -> LazyGroupBy:
        return LazyGroupBy(self._df, self._plan, by)

    def order_by(self, by: Any, descending: bool = False) -> LazyDF:
        return LazyDF(
            self._df, self._plan + [("order_by", (by,), {"descending": descending})]
        )

    def head(self, n: int = 5) -> LazyDF:
        # Optimization: if head is the first operation, we can slice early
        return LazyDF(self._df, self._plan + [("head", (n,), {})])

    def collect(self) -> Any:
        """Execute the recorded query plan."""
        from .df_wrapper import DF

        df = self._df
        for op, args, kwargs in self._plan:
            wrapper = DF(df)
            if op == "filter":
                df = wrapper[args[0]].raw
            elif op == "mutate":
                df = wrapper.mutate(**kwargs).raw
            elif op == "select":
                df = wrapper.select(*args).raw
            elif op == "order_by":
                df = wrapper.order_by(*args, **kwargs).raw
            elif op == "head":
                df = wrapper.head(*args).raw
            elif op == "groupby_agg":
                df = wrapper.groupby(args[0]).agg(**kwargs).raw
            else:
                raise NotImplementedError(f"Operation {op} not implemented in LazyDF.")

        return DF(df)

    def __repr__(self) -> str:
        return f"LazyDF(plan={self._plan}, source={self._df!r})"


class LazyGroupBy:
    def __init__(self, df: Any, plan: list, by: Any) -> None:
        self._df = df
        self._plan = plan
        self._by = by

    def agg(self, **kwargs: Any) -> LazyDF:
        return LazyDF(self._df, self._plan + [("groupby_agg", (self._by,), kwargs)])
