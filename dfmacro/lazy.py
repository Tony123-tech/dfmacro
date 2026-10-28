"""Lazy evaluation engine for dfmacro."""

from __future__ import annotations

from typing import Any

from .expr import Expr
from .optimizer import optimize_plan
from .resilience import retry


class LazyDF:
    """Lazy DataFrame plan that can represent branches and multi-table joins.

    Join nodes retain the right-hand ``LazyDF`` as a child plan, so a sequence
    of joins forms a directed acyclic graph rather than a flattened list of
    operations against one source.
    """

    __slots__ = ("_df", "_plan")

    def __init__(
        self, df: Any, plan: list[tuple[str, tuple, dict]] | None = None
    ) -> None:
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_plan", plan or [])

    def __getitem__(self, key: Any) -> LazyDF:
        if isinstance(key, Expr):
            return self.filter(key)
        raise TypeError(
            "LazyDF only supports expression filtering (e.g. df[price > 100]) for now."
        )

    def filter(self, predicate: Expr) -> LazyDF:
        """Record a row filter.

        Args:
            predicate: Boolean expression evaluated against the current rows.

        Returns:
            A new lazy plan with the filter appended.
        """
        if not isinstance(predicate, Expr):
            raise TypeError("LazyDF.filter expects a dfmacro Expr predicate.")
        return LazyDF(self._df, self._plan + [("filter", (predicate,), {})])

    def join(
        self, other_lazy_df: LazyDF, on: Any = "col", how: str = "inner"
    ) -> LazyDF:
        """Record a join with another lazy plan, preserving both branches.

        Args:
            other_lazy_df: Right-hand ``LazyDF`` branch.
            on: Shared join column name or a sequence of names.
            how: Join type supported by the backend, normally ``"inner"``,
                ``"left"``, ``"right"``, or ``"outer"``.

        Returns:
            A new lazy plan containing a DAG join node.
        """
        if not isinstance(other_lazy_df, LazyDF):
            raise TypeError("LazyDF.join expects another LazyDF instance.")
        if how not in {"inner", "left", "right", "outer", "full"}:
            raise ValueError(
                "how must be one of 'inner', 'left', 'right', 'outer', or 'full'."
            )
        return LazyDF(
            self._df,
            self._plan + [("join", (other_lazy_df,), {"on": on, "how": how})],
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
        return LazyDF(self._df, self._plan + [("head", (n,), {})])

    def tail(self, n: int = 5) -> LazyDF:
        return LazyDF(self._df, self._plan + [("tail", (n,), {})])

    def rename(self, **mapping: str) -> LazyDF:
        """Record column renames for execution and lineage reporting."""
        return LazyDF(self._df, self._plan + [("rename", (), {"mapping": mapping})])

    def explain(self) -> str:
        """Return a numbered plan tree with simple column lineage information."""
        lines: list[str] = []
        current_columns = list(getattr(self._df, "columns", ()))
        for index, (operation, args, kwargs) in enumerate(self._plan, start=1):
            if operation == "filter":
                detail = f"Filter ({args[0]!r})"
            elif operation == "mutate":
                detail = "Mutate (" + ", ".join(kwargs) + ")"
                for name in kwargs:
                    if name not in current_columns:
                        current_columns.append(name)
            elif operation == "select":
                detail = "Select (" + ", ".join(map(str, args)) + ")"
                current_columns = [
                    value.name if hasattr(value, "name") else str(value)
                    for value in args
                ]
            elif operation == "join":
                other = args[0]
                detail = f"Join ({kwargs.get('how', 'inner')}, on: {kwargs.get('on')})"
                lines.append(f"{index}. {detail}")
                branch = other.explain().splitlines()
                lines.extend(f"   right -> {line}" for line in branch)
                other_columns = list(getattr(other._df, "columns", ()))
                join_keys = kwargs.get("on")
                join_keys = (
                    list(join_keys)
                    if isinstance(join_keys, (tuple, list))
                    else [join_keys]
                )
                current_columns.extend(
                    column for column in other_columns if column not in join_keys
                )
                continue
            elif operation == "groupby_agg":
                detail = f"GroupBy ({args[0]}) -> Aggregate ({', '.join(kwargs)})"
            elif operation == "order_by":
                detail = f"OrderBy ({args[0]})"
            elif operation in {"head", "tail"}:
                detail = f"{operation.title()} ({args[0]})"
            elif operation == "rename":
                mapping = kwargs.get("mapping", {})
                lineage = ", ".join(f"{old} -> {new}" for old, new in mapping.items())
                detail = f"Rename ({lineage})"
                current_columns = [mapping.get(column, column) for column in current_columns]
            else:
                detail = operation.title()
            lines.append(f"{index}. {detail}")
        if current_columns:
            lines.append("Columns: " + ", ".join(map(str, current_columns)))
        return "\n".join(lines) if lines else "Source DataFrame (no operations)"

    @retry()
    def collect(self) -> Any:
        """Execute the recorded query plan."""
        from .df_wrapper import DF

        if type(self._df).__module__.startswith("polars"):
            import polars as pl

            from .compiler.polars import compile_lazy_plan

            source = self._df.lazy() if isinstance(self._df, pl.DataFrame) else self._df
            frame = compile_lazy_plan(optimize_plan(self._plan), source).collect()
            return DF(frame)

        df = self._df
        for op, args, kwargs in optimize_plan(self._plan):
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
            elif op == "tail":
                df = wrapper.tail(*args).raw
            elif op == "groupby_agg":
                if type(df).__module__.startswith("dask.dataframe"):
                    from .compiler.dask import execute_groupby_agg

                    df = execute_groupby_agg(df, args[0], kwargs)
                else:
                    df = wrapper.groupby(args[0]).agg(**kwargs).raw
            elif op == "join":
                other_plan = args[0]
                other_df = other_plan.collect().raw
                if type(df).__module__.startswith("dask.dataframe"):
                    from .compiler.dask import execute_join

                    df = execute_join(df, other_df, kwargs["on"], kwargs["how"])
                elif type(df).__module__.startswith("pandas"):
                    from .compiler.pandas import execute_join

                    df = execute_join(df, other_df, kwargs["on"], kwargs["how"])
                else:
                    df = wrapper.merge(
                        other_df, on=kwargs["on"], how=kwargs["how"]
                    ).raw
            elif op == "rename":
                df = wrapper.rename(**kwargs.get("mapping", {})).raw
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
