"""SQL backend for dfmacro expression compilation."""

from __future__ import annotations

from typing import Any

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
        return f'"{expr.name}"'

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
        if expr.method == "contains":
            pat = expr.kwargs["pat"].replace("'", "''")
            return f"{col} LIKE '%{pat}%'"
        if expr.method == "startswith":
            pat = expr.kwargs["pat"].replace("'", "''")
            return f"{col} LIKE '{pat}%'"
        if expr.method == "endswith":
            pat = expr.kwargs["pat"].replace("'", "''")
            return f"{col} LIKE '%{pat}'"
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
            return f"EXTRACT(DOW FROM {col})"
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
