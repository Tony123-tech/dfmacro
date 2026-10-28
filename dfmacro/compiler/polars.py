"""Polars backend for dfmacro expression compilation."""

from __future__ import annotations

import difflib
from typing import Any, Callable

import polars as pl

from ..expr import (
    AggExpr,
    AccessorExpr,
    BinaryExpr,
    ColumnExpr,
    Expr,
    Literal,
    MathExpr,
    UnaryExpr,
    WhenExpr,
    WindowExpr,
)
from . import CompileError
from .accessor import compile_accessor_method

_BINARY_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
}


def compile_expr(expr: Expr, df: Any) -> Any:
    if isinstance(expr, AccessorExpr):
        col = compile_expr(expr.column, df)
        return compile_accessor_method(expr, col, "polars")

    if isinstance(expr, ColumnExpr):
        available = (
            df.collect_schema().names()
            if type(df).__name__ == "LazyFrame"
            else df.columns
        )
        if expr.name not in available:
            matches = difflib.get_close_matches(expr.name, available, n=1, cutoff=0.6)
            suggestion = f" Did you mean '{matches[0]}'?" if matches else ""

            raise CompileError(
                f"Column {expr.name!r} not found. Available: {available}.{suggestion}"
            )
        return pl.col(expr.name)

    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, BinaryExpr):
        left = compile_expr(expr.left, df)
        right = compile_expr(expr.right, df)
        op_func = _BINARY_OPS.get(expr.op)
        if op_func is None:
            raise CompileError(f"Unsupported operator: {expr.op!r}")
        try:
            return op_func(left, right)
        except Exception as exc:
            raise CompileError(f"Failed to apply {expr.op!r}: {exc}") from exc

    if isinstance(expr, UnaryExpr):
        operand = compile_expr(expr.operand, df)
        if expr.op == "~":
            return ~operand
        raise CompileError(f"Unsupported unary op: {expr.op!r}")

    if isinstance(expr, WhenExpr):
        cond = compile_expr(expr.condition, df)
        then_val = compile_expr(expr.then_value, df)
        else_val = (
            compile_expr(expr.else_value, df) if expr.else_value is not None else None
        )
        if else_val is not None:
            return pl.when(cond).then(then_val).otherwise(else_val)
        return pl.when(cond).then(then_val)

    if isinstance(expr, WindowExpr):
        col = compile_expr(expr.column, df)
        if expr.func == "rank":
            return col.rank(
                method=expr.kwargs["method"], descending=expr.kwargs["descending"]
            )
        elif expr.func == "cumsum":
            return col.cum_sum()
        elif expr.func == "cummax":
            return col.cum_max()
        elif expr.func == "cummin":
            return col.cum_min()
        elif expr.func == "shift":
            return col.shift(expr.kwargs["periods"])
        elif expr.func == "diff":
            return col.diff()
        elif expr.func == "pct_change":
            return col.pct_change()
        raise CompileError(f"Unsupported window func: {expr.func}")

    if isinstance(expr, MathExpr):
        col = compile_expr(expr.column, df)
        if expr.func == "abs":
            return col.abs()
        elif expr.func == "round":
            return col.round(expr.kwargs["n"])
        elif expr.func == "sqrt":
            return col.sqrt()
        elif expr.func == "pow":
            return col.pow(expr.kwargs["n"])
        raise CompileError(f"Unsupported math func: {expr.func}")

    raise CompileError(f"Unknown node: {type(expr).__name__}")


def compile_lazy_plan(plan: list[tuple[str, tuple, dict]], lazy_frame: Any) -> Any:
    """Lower a dfmacro plan directly into a native Polars LazyFrame plan."""
    frame = lazy_frame
    for operation, args, kwargs in plan:
        if operation == "filter":
            frame = frame.filter(compile_expr(args[0], frame))
        elif operation == "mutate":
            expressions = []
            for name, value in kwargs.items():
                compiled = compile_expr(value, frame) if isinstance(value, Expr) else pl.lit(value)
                expressions.append(compiled.alias(name))
            frame = frame.with_columns(expressions)
        elif operation == "select":
            expressions = [
                compile_expr(value, frame) if isinstance(value, Expr) else pl.col(value)
                for value in args
            ]
            frame = frame.select(expressions)
        elif operation == "order_by":
            column = args[0]
            name = column.name if isinstance(column, ColumnExpr) else column
            frame = frame.sort(name, descending=kwargs.get("descending", False))
        elif operation == "head":
            frame = frame.head(args[0])
        elif operation == "tail":
            frame = frame.tail(args[0])
        elif operation == "join":
            other_plan = args[0]
            other_source = other_plan._df
            other_frame = other_source.lazy() if isinstance(other_source, pl.DataFrame) else other_source
            other_frame = compile_lazy_plan(other_plan._plan, other_frame)
            frame = frame.join(
                other_frame,
                on=kwargs["on"],
                how="full" if kwargs["how"] in {"outer", "full"} else kwargs["how"],
            )
        elif operation == "groupby_agg":
            by = args[0]
            group_columns = [by] if not isinstance(by, (tuple, list)) else list(by)
            aggregations = []
            for name, expression in kwargs.items():
                if not isinstance(expression, AggExpr):
                    raise CompileError(
                        f"Unsupported Polars aggregation for {name!r}: "
                        f"{type(expression).__name__}."
                    )
                aggregate = getattr(pl.col(expression.column.name), expression.func)(
                    **expression.kwargs
                )
                aggregations.append(aggregate.alias(name))
            frame = frame.group_by(group_columns).agg(aggregations)
        elif operation == "rename":
            mapping = kwargs.get("mapping", {})
            frame = frame.rename(mapping)
        else:
            raise CompileError(
                f"Operation {operation!r} is not supported by the Polars lazy compiler."
            )
    return frame
