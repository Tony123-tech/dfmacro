"""SQL backend for dfmacro expression compilation."""

from __future__ import annotations

from typing import Any, Sequence

from ..expr import (
    AggExpr,
    BinaryExpr,
    ColumnExpr,
    DatePartExpr,
    Expr,
    Literal,
    MathExpr,
    StringMethodExpr,
    UnaryExpr,
    WhenExpr,
)
from ..optimizer import optimize_plan
from . import CompileError

_BINARY_OPS = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
    "==": "=",
    "!=": "!=",
    "&": "AND",
    "|": "OR",
}

# Aggregate function name -> DuckDB SQL function.
# Functions that need special argument handling (nunique, quantile) are
# handled explicitly in _compile_agg below.
_AGG_SQL = {
    "mean": "AVG",
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
    "count": "COUNT",
    "median": "MEDIAN",
    "std": "STDDEV_SAMP",
    "var": "VAR_SAMP",
    "first": "FIRST",
    "last": "LAST",
}


def compile_expr(expr: Expr, df: Any = None) -> str:
    if isinstance(expr, ColumnExpr):
        return _quote_identifier(expr.name)

    if isinstance(expr, Literal):
        if isinstance(expr.value, str):
            # Escape single quotes for SQL string literals.
            escaped = expr.value.replace("'", "''")
            return f"'{escaped}'"
        if isinstance(expr.value, bool):
            return "TRUE" if expr.value else "FALSE"
        if expr.value is None:
            return "NULL"
        return str(expr.value)

    if isinstance(expr, BinaryExpr):
        left = compile_expr(expr.left)
        right = compile_expr(expr.right)
        op = _BINARY_OPS.get(expr.op)
        if op is None:
            raise CompileError(f"Unsupported SQL operator: {expr.op!r}")
        return f"({left} {op} {right})"

    if isinstance(expr, UnaryExpr):
        operand = compile_expr(expr.operand)
        if expr.op == "~":
            return f"NOT ({operand})"
        raise CompileError(f"Unsupported unary op: {expr.op!r}")

    if isinstance(expr, StringMethodExpr):
        col = compile_expr(expr.column)
        if expr.method in ("contains", "startswith", "endswith"):
            raw = expr.kwargs["pat"]
            if expr.kwargs.get("regex", False):
                escaped = raw.replace("'", "''")
                return f"REGEXP_MATCHES({col}, '{escaped}')"
            escaped = (
                raw.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
                .replace("'", "''")
            )
            if expr.method == "contains":
                return f"{col} LIKE '%{escaped}%' ESCAPE '\\'"
            if expr.method == "startswith":
                return f"{col} LIKE '{escaped}%' ESCAPE '\\'"
            return f"{col} LIKE '%{escaped}' ESCAPE '\\'"
        if expr.method == "lower":
            return f"LOWER({col})"
        if expr.method == "upper":
            return f"UPPER({col})"
        if expr.method == "strip":
            return f"TRIM({col})"
        if expr.method == "len":
            return f"LENGTH({col})"
        if expr.method == "replace":
            pat = expr.kwargs["pat"].replace("'", "''")
            repl = expr.kwargs["repl"].replace("'", "''")
            return f"REPLACE({col}, '{pat}', '{repl}')"
        if expr.method == "slice":
            start = expr.kwargs["start"] + 1  # SQL is 1-indexed
            length = expr.kwargs["length"]
            return f"SUBSTRING({col}, {start}, {length})"
        raise CompileError(f"Unsupported SQL string method: {expr.method}")

    if isinstance(expr, DatePartExpr):
        col = compile_expr(expr.column)
        if expr.part == "year":
            return f"EXTRACT(YEAR FROM {col})"
        if expr.part == "month":
            return f"EXTRACT(MONTH FROM {col})"
        if expr.part == "day":
            return f"EXTRACT(DAY FROM {col})"
        if expr.part == "hour":
            return f"EXTRACT(HOUR FROM {col})"
        if expr.part == "minute":
            return f"EXTRACT(MINUTE FROM {col})"
        if expr.part == "dayofweek":
            return f"(EXTRACT(ISODOW FROM {col}) - 1)"
        if expr.part == "quarter":
            return f"EXTRACT(QUARTER FROM {col})"
        if expr.part == "strftime":
            fmt = expr.kwargs["fmt"].replace("'", "''")
            return f"STRFTIME({col}, '{fmt}')"
        raise CompileError(f"Unsupported SQL date part: {expr.part}")

    if isinstance(expr, AggExpr):
        return _compile_agg(expr)

    if isinstance(expr, MathExpr):
        col = compile_expr(expr.column)
        if expr.func == "abs":
            return f"ABS({col})"
        if expr.func == "round":
            return f"ROUND({col}, {expr.kwargs['n']})"
        if expr.func == "sqrt":
            return f"SQRT({col})"
        if expr.func == "pow":
            return f"POWER({col}, {expr.kwargs['n']})"
        raise CompileError(f"Unsupported SQL math func: {expr.func!r}")

    if isinstance(expr, WhenExpr):
        cond = compile_expr(expr.condition)
        then_val = compile_expr(expr.then_value)
        if expr.else_value is not None:
            else_val = compile_expr(expr.else_value)
            return f"CASE WHEN {cond} THEN {then_val} ELSE {else_val} END"
        return f"CASE WHEN {cond} THEN {then_val} END"

    raise CompileError(f"Unknown expression node type: {type(expr).__name__}")


def _quote_identifier(identifier: Any) -> str:
    """Quote one SQL identifier, escaping embedded quote characters."""
    if not isinstance(identifier, str) or not identifier:
        raise CompileError(
            f"SQL identifiers must be non-empty strings, got {identifier!r}."
        )
    return '"' + identifier.replace('"', '""') + '"'


def _column_sql(column: Any) -> str:
    if isinstance(column, Expr):
        return compile_expr(column)
    if hasattr(column, "name") and isinstance(column.name, str):
        column = column.name
    return _quote_identifier(column)


def _value_sql(value: Any) -> str:
    return compile_expr(value if isinstance(value, Expr) else Literal(value))


def compile_query(
    plan: Sequence[tuple[str, tuple, dict]],
    table_name: str = "df",
    columns: Sequence[str] | None = None,
) -> str:
    """Compile a LazyDF DAG into readable DuckDB SQL using named CTEs."""
    plan = optimize_plan(list(plan))
    if not isinstance(table_name, str) or not table_name:
        raise CompileError("table_name must be a non-empty SQL table identifier.")
    ctes: list[tuple[str, str]] = []
    state = {"cte": 0, "table": 2}
    has_join = any(operation == "join" for operation, _, _ in plan)
    if has_join and table_name == "df":
        table_name = "df1"
    final_name, known_columns, order_terms = _compile_plan_ctes(
        plan,
        table_name,
        list(columns) if columns is not None else None,
        ctes,
        state,
    )
    with_clause = "WITH\n" + ",\n".join(
        f"  {_quote_identifier(name)} AS (\n"
        + "\n".join(f"    {line}" for line in query.splitlines())
        + "\n  )"
        for name, query in ctes
    )
    final_query = f"SELECT * FROM {_quote_identifier(final_name)}"
    if order_terms:
        final_query += f"\nORDER BY {_format_order(order_terms)}"
    return f"{with_clause}\n{final_query}"


def _compile_plan_ctes(
    plan: Sequence[tuple[str, tuple, dict]],
    table_name: str,
    columns: Sequence[str] | None,
    ctes: list[tuple[str, str]],
    state: dict[str, int],
) -> tuple[str, list[str] | None, list[tuple[str, bool]]]:
    """Compile one plan branch into named CTEs and return its final relation."""
    source = ".".join(_quote_identifier(part) for part in table_name.split("."))
    current = _add_cte(ctes, state, f"SELECT * FROM {source}")
    known_columns = list(columns) if columns is not None else None
    order_terms: list[tuple[str, bool]] = []
    source_alias = table_name.rsplit(".", 1)[-1]

    for operation, args, kwargs in plan:
        relation = _quote_identifier(current)
        if operation == "filter":
            query = f"SELECT * FROM {relation} " f"WHERE {compile_expr(args[0])}"
        elif operation == "mutate":
            updates = dict(kwargs)
            if known_columns is None:
                projections = ["*"] + [
                    f"{_value_sql(value)} AS {_quote_identifier(name)}"
                    for name, value in updates.items()
                ]
            else:
                known_columns = list(known_columns)
                for name in updates:
                    if name not in known_columns:
                        known_columns.append(name)
                projections = [
                    (
                        f"{_value_sql(updates[name])} AS {_quote_identifier(name)}"
                        if name in updates
                        else _quote_identifier(name)
                    )
                    for name in known_columns
                ]
            query = f"SELECT {', '.join(projections)} FROM {relation}"
        elif operation == "select":
            if not args:
                raise CompileError("SQL select requires at least one column.")
            selected = [_column_sql(value) for value in args]
            query = f"SELECT {', '.join(selected)} FROM {relation}"
            known_columns = [
                value.name if isinstance(value, ColumnExpr) else str(value)
                for value in args
            ]
        elif operation == "join":
            other = args[0]
            from ..lazy import LazyDF

            if not isinstance(other, LazyDF):
                raise CompileError("SQL join plans must contain a LazyDF branch.")
            join_keys = kwargs.get("on")
            join_keys = (
                list(join_keys) if isinstance(join_keys, (list, tuple)) else [join_keys]
            )
            if not join_keys or any(not isinstance(key, str) for key in join_keys):
                raise CompileError("SQL joins require one or more string column names.")
            if known_columns is not None and not set(join_keys).issubset(known_columns):
                raise CompileError(
                    f"SQL join keys {join_keys!r} are unavailable in the left plan."
                )
            right_alias = f"df{state['table']}"
            state["table"] += 1
            right_columns = getattr(other._df, "columns", None)
            right_final, right_known, _ = _compile_plan_ctes(
                optimize_plan(list(other._plan)),
                right_alias,
                list(right_columns) if right_columns is not None else None,
                ctes,
                state,
            )
            if right_known is not None and not set(join_keys).issubset(right_known):
                raise CompileError(
                    f"SQL join keys {join_keys!r} are unavailable in the right plan."
                )
            how = kwargs.get("how", "inner")
            join_type = {
                "inner": "INNER",
                "left": "LEFT",
                "right": "RIGHT",
                "outer": "FULL OUTER",
                "full": "FULL OUTER",
            }.get(how)
            if join_type is None:
                raise CompileError(f"Unsupported SQL join type: {how!r}.")
            conditions = " AND ".join(
                f"{_quote_identifier(source_alias)}.{_quote_identifier(key)} = "
                f"{_quote_identifier(right_alias)}.{_quote_identifier(key)}"
                for key in join_keys
            )
            if known_columns is not None and right_known is not None:
                left_names = list(known_columns)
                right_names = list(right_known)
                overlaps = (set(left_names) & set(right_names)) - set(join_keys)
                if overlaps or how in {"right", "outer", "full"}:
                    projections = []
                    for name in left_names:
                        if name in join_keys and how in {"right", "outer", "full"}:
                            projections.append(
                                f"COALESCE({_quote_identifier(right_alias)}."
                                f"{_quote_identifier(name)}, "
                                f"{_quote_identifier(source_alias)}."
                                f"{_quote_identifier(name)}) AS {_quote_identifier(name)}"
                            )
                        elif name in overlaps:
                            projections.append(
                                f"{_quote_identifier(source_alias)}."
                                f"{_quote_identifier(name)} AS "
                                f"{_quote_identifier(name + '_x')}"
                            )
                        else:
                            projections.append(
                                f"{_quote_identifier(source_alias)}."
                                f"{_quote_identifier(name)}"
                            )
                    for name in right_names:
                        if name in join_keys:
                            continue
                        output_name = name + "_y" if name in overlaps else name
                        projections.append(
                            f"{_quote_identifier(right_alias)}."
                            f"{_quote_identifier(name)} AS "
                            f"{_quote_identifier(output_name)}"
                        )
                    query = (
                        f"SELECT {', '.join(projections)}\n"
                        f"FROM {relation} AS {_quote_identifier(source_alias)}\n"
                        f"{join_type} JOIN {_quote_identifier(right_final)} "
                        f"AS {_quote_identifier(right_alias)} ON {conditions}"
                    )
                    known_columns = [
                        name + "_x" if name in overlaps else name for name in left_names
                    ] + [
                        name + "_y" if name in overlaps else name
                        for name in right_names
                        if name not in join_keys
                    ]
                else:
                    exclude_keys = ", ".join(
                        _quote_identifier(key) for key in join_keys
                    )
                    query = (
                        f"SELECT {_quote_identifier(source_alias)}.*, "
                        f"{_quote_identifier(right_alias)}.* EXCLUDE ({exclude_keys})\n"
                        f"FROM {relation} AS {_quote_identifier(source_alias)}\n"
                        f"{join_type} JOIN {_quote_identifier(right_final)} "
                        f"AS {_quote_identifier(right_alias)} ON {conditions}"
                    )
                    known_columns = left_names + [
                        name for name in right_names if name not in join_keys
                    ]
            else:
                exclude_keys = ", ".join(_quote_identifier(key) for key in join_keys)
                query = (
                    f"SELECT {_quote_identifier(source_alias)}.*, "
                    f"{_quote_identifier(right_alias)}.* EXCLUDE ({exclude_keys})\n"
                    f"FROM {relation} AS {_quote_identifier(source_alias)}\n"
                    f"{join_type} JOIN {_quote_identifier(right_final)} "
                    f"AS {_quote_identifier(right_alias)} ON {conditions}"
                )
                known_columns = None
            order_terms = []
            source_alias = current
        elif operation in {"groupby", "groupby_agg"}:
            by = args[0]
            group_columns = [by] if not isinstance(by, (list, tuple)) else list(by)
            group_sql = [_column_sql(column) for column in group_columns]
            projections = list(group_sql)
            for name, expression in kwargs.items():
                if not isinstance(expression, AggExpr):
                    raise CompileError(
                        f"Unsupported SQL aggregation for {name!r}: "
                        f"{type(expression).__name__}."
                    )
                projections.append(
                    f"{_compile_agg(expression)} AS {_quote_identifier(name)}"
                )
            query = (
                f"SELECT {', '.join(projections)} FROM {relation} "
                f"GROUP BY {', '.join(group_sql)}"
            )
            known_columns = [
                *(
                    column.name if isinstance(column, ColumnExpr) else str(column)
                    for column in group_columns
                ),
                *kwargs.keys(),
            ]
        elif operation == "order_by":
            column = args[0]
            column_name = column.name if isinstance(column, ColumnExpr) else str(column)
            if known_columns is not None and column_name not in known_columns:
                raise CompileError(
                    f"SQL order_by column {column_name!r} is unavailable."
                )
            descending = bool(kwargs.get("descending", False))
            order_terms = [(_column_sql(column), descending)]
            query = f"SELECT * FROM {relation} ORDER BY {_format_order(order_terms)}"
        elif operation in {"head", "tail"}:
            count = args[0]
            if not isinstance(count, int):
                raise CompileError(f"SQL {operation} count must be an integer.")
            if operation == "tail" and not order_terms:
                raise CompileError(
                    "SQL tail requires a preceding order_by because SQL rows "
                    "have no intrinsic order."
                )
            if operation == "head":
                if count < 0 and not order_terms:
                    raise CompileError(
                        "SQL head with a negative count requires a preceding order_by."
                    )
                if count < 0:
                    reverse = _format_order(order_terms, reverse=True)
                    query = (
                        f"SELECT * FROM (SELECT * FROM {relation} "
                        f"ORDER BY {reverse} LIMIT {-count}) AS _dfmacro_reverse "
                        f"ORDER BY {_format_order(order_terms)}"
                    )
                else:
                    query = f"SELECT * FROM {relation}"
                    if order_terms:
                        query += f" ORDER BY {_format_order(order_terms)}"
                    query += f" LIMIT {count}"
            else:
                if count < 0:
                    query = (
                        f"SELECT * FROM {relation} ORDER BY "
                        f"{_format_order(order_terms)} OFFSET {-count}"
                    )
                else:
                    query = (
                        f"SELECT * FROM (SELECT * FROM {relation} ORDER BY "
                        f"{_format_order(order_terms, reverse=True)} LIMIT {count}) "
                        f"AS _dfmacro_tail ORDER BY {_format_order(order_terms)}"
                    )
        elif operation == "rename":
            mapping = kwargs.get("mapping", {})
            projections = [
                f"{_quote_identifier(name)} AS "
                f"{_quote_identifier(mapping.get(name, name))}"
                for name in (known_columns or ())
            ]
            if not projections:
                raise CompileError("SQL rename requires known source columns.")
            query = f"SELECT {', '.join(projections)} FROM {relation}"
            if known_columns is not None:
                known_columns = [mapping.get(name, name) for name in known_columns]
        else:
            raise CompileError(f"Unsupported SQL pipeline operation: {operation!r}.")
        current = _add_cte(ctes, state, query)
    return current, known_columns, order_terms


def _add_cte(ctes: list[tuple[str, str]], state: dict[str, int], query: str) -> str:
    state["cte"] += 1
    name = f"_dfmacro_{state['cte']}"
    ctes.append((name, query))
    return name


def _format_order(
    order_terms: Sequence[tuple[str, bool]], reverse: bool = False
) -> str:
    return ", ".join(
        f"{column} {'ASC' if descending == reverse else 'DESC'}"
        for column, descending in order_terms
    )


def _compile_agg(expr: AggExpr) -> str:
    """Compile an AggExpr to a DuckDB aggregate function call."""
    col = compile_expr(expr.column)

    # Functions with non-standard SQL syntax
    if expr.func == "nunique":
        return f"COUNT(DISTINCT {col})"
    if expr.func == "quantile":
        q = expr.kwargs["q"]
        return f"QUANTILE_CONT({col}, {q})"

    sql_func = _AGG_SQL.get(expr.func)
    if sql_func is None:
        raise CompileError(f"Unsupported SQL aggregate: {expr.func!r}")
    return f"{sql_func}({col})"
