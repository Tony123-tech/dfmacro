"""SQL backend for dfmacro expression compilation."""

from __future__ import annotations

from typing import Any

from ..expr import (
    BinaryExpr,
    ColumnExpr,
    DatePartExpr,
    Expr,
    Literal,
    StringMethodExpr,
    UnaryExpr,
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


def compile_expr(expr: Expr, df: Any = None) -> str:
    if isinstance(expr, ColumnExpr):
        return f'"{expr.name}"'
    if isinstance(expr, Literal):
        if isinstance(expr.value, str):
            return f"'{expr.value}'"
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
            return f"{col} LIKE '%{expr.kwargs['pat']}%'"
        raise CompileError(f"Unsupported SQL string method: {expr.method}")
    if isinstance(expr, DatePartExpr):
        col = compile_expr(expr.column)
        if expr.part == "year":
            return f"EXTRACT(YEAR FROM {col})"
        elif expr.part == "month":
            return f"EXTRACT(MONTH FROM {col})"
        raise CompileError(f"Unsupported SQL date part: {expr.part}")
    raise CompileError(f"Unknown expression node type: {type(expr).__name__}")
