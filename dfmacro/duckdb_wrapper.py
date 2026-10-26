"""DuckDB backend for dfmacro."""

from __future__ import annotations

from typing import Any

import duckdb

from .compiler.sql import compile_expr
from .expr import AggExpr, Expr


class DuckDBGroupBy:
    def __init__(self, df: Any, by: Any, view_name: str, conn: Any) -> None:
        self._df = df
        self._by = by.name if hasattr(by, "name") else by
        self._view_name = view_name
        self._conn = conn

    def agg(self, **kwargs: Any) -> Any:
        select_parts = [f'"{self._by}"']
        for out_col, expr in kwargs.items():
            if isinstance(expr, AggExpr):
                select_parts.append(f'{compile_expr(expr)} AS "{out_col}"')
            else:
                raise TypeError(f"Unsupported agg expression: {expr}")

        query = (
            f'SELECT {", ".join(select_parts)} '
            f"FROM {self._view_name} "
            f'GROUP BY "{self._by}"'
        )
        result = self._conn.execute(query).df()
        return DuckDBDF(result, "result_df", self._conn)


class DuckDBDF:
    """A wrapper that executes dfmacro expressions using DuckDB SQL engine (Eager)."""

    __slots__ = ("_conn", "_df", "_view_name")

    def __init__(self, df: Any, view_name: str = "df", conn: Any = None) -> None:
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_view_name", view_name)

        if conn is None:
            conn = duckdb.connect(database=":memory:")
            conn.register(view_name, df)

        object.__setattr__(self, "_conn", conn)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, Expr):
            sql_where = compile_expr(key)
            query = f"SELECT * FROM {self._view_name} WHERE {sql_where}"
            result = self._conn.execute(query).df()
            return DuckDBDF(result, "result_df", self._conn)
        raise TypeError("DuckDBDF only supports expression filtering.")

    def groupby(self, by: Any) -> DuckDBGroupBy:
        return DuckDBGroupBy(self._df, by, self._view_name, self._conn)

    def mutate(self, **kwargs: Any) -> Any:
        select_parts = ["*"]
        for out_col, expr in kwargs.items():
            sql_expr = compile_expr(expr)
            select_parts.append(f'{sql_expr} AS "{out_col}"')

        query = f"SELECT {', '.join(select_parts)} FROM {self._view_name}"
        result = self._conn.execute(query).df()
        return DuckDBDF(result, "result_df", self._conn)

    def join(self, other: Any, on: Any, how: str = "inner") -> Any:
        on_col = on.name if hasattr(on, "name") else on
        other_view = "other_df"
        other_df = other._df if isinstance(other, DuckDBDF) else other
        self._conn.register(other_view, other_df)

        join_type = "INNER" if how == "inner" else how.upper()
        query = (
            f"SELECT * FROM {self._view_name} a "
            f"{join_type} JOIN {other_view} b USING ({on_col})"
        )
        result = self._conn.execute(query).df()
        return DuckDBDF(result, "result_df", self._conn)

    def lazy(self) -> LazyDuckDBDF:
        """Switch to lazy evaluation mode."""
        return LazyDuckDBDF(self._df, self._view_name, self._conn)

    def collect(self) -> Any:
        return self._df

    def __repr__(self) -> str:
        return f"DuckDBDF(view={self._view_name}, df={self._df!r})"


class LazyDuckDBGroupBy:
    def __init__(self, lazy_df: LazyDuckDBDF, by: Any) -> None:
        self._lazy_df = lazy_df
        self._by = by.name if hasattr(by, "name") else by

    def agg(self, **kwargs: Any) -> LazyDuckDBDF:
        return self._lazy_df._add_op("groupby", (self._by,), kwargs)


class LazyDuckDBDF:
    """Lazy wrapper for DuckDB that collects SQL operations into a plan."""

    __slots__ = ("_conn", "_df", "_plan", "_view_name")

    def __init__(
        self, df: Any, view_name: str = "df", conn: Any = None, plan: list | None = None
    ) -> None:
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_view_name", view_name)
        object.__setattr__(self, "_plan", plan or [])

        if conn is None:
            conn = duckdb.connect(database=":memory:")
            conn.register(view_name, df)
        object.__setattr__(self, "_conn", conn)

    def _add_op(self, op: str, args: tuple, kwargs: dict) -> LazyDuckDBDF:
        return LazyDuckDBDF(
            self._df, self._view_name, self._conn, self._plan + [(op, args, kwargs)]
        )

    def __getitem__(self, key: Any) -> LazyDuckDBDF:
        if isinstance(key, Expr):
            return self._add_op("filter", (key,), {})
        raise TypeError("LazyDuckDBDF only supports expression filtering.")

    def mutate(self, **kwargs: Any) -> LazyDuckDBDF:
        return self._add_op("mutate", (), kwargs)

    def groupby(self, by: Any) -> LazyDuckDBGroupBy:
        return LazyDuckDBGroupBy(self, by)

    def collect(self) -> DuckDBDF:
        """Compile the plan into a single SQL query and execute it."""
        select_parts = ["*"]
        where_parts: list[str] = []
        groupby_part = ""

        for op, args, kwargs in self._plan:
            if op == "filter":
                where_parts.append(f"({compile_expr(args[0])})")
            elif op == "mutate":
                for out_col, expr in kwargs.items():
                    sql_expr = compile_expr(expr)
                    select_parts.append(f'{sql_expr} AS "{out_col}"')
            elif op == "groupby":
                by_col = args[0]
                # Aggregations replace the select list entirely.
                select_parts = [f'"{by_col}"']
                for out_col, expr in kwargs.items():
                    if isinstance(expr, AggExpr):
                        select_parts.append(f'{compile_expr(expr)} AS "{out_col}"')
                    else:
                        raise TypeError(
                            f"Unsupported agg expression in groupby: {expr}"
                        )
                groupby_part = f' GROUP BY "{by_col}"'

        query = f"SELECT {', '.join(select_parts)} FROM {self._view_name}"
        if where_parts:
            query += f" WHERE {' AND '.join(where_parts)}"
        query += groupby_part

        result = self._conn.execute(query).df()
        return DuckDBDF(result, "result_df", self._conn)

    def __repr__(self) -> str:
        return f"LazyDuckDBDF(plan={self._plan})"
