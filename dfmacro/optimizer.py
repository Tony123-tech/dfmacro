"""Conservative expression and lazy-plan optimizations."""

from __future__ import annotations

import operator
from typing import Any

from .expr import (
    AccessorExpr,
    AggExpr,
    BinaryExpr,
    ColumnExpr,
    Expr,
    Literal,
    MathExpr,
    DatePartExpr,
    StringMethodExpr,
    UnaryExpr,
    WhenExpr,
    WindowExpr,
)

_FOLD_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "&": operator.and_,
    "|": operator.or_,
}
_SAFE_LITERAL_TYPES = (bool, int, float, str, bytes, type(None))


def _safe_literal(value: Any) -> bool:
    return type(value) in _SAFE_LITERAL_TYPES


def optimize_expr(expr: Expr) -> Expr:
    """Return an equivalent expression with safe literal subtrees folded.

    Only built-in scalar literals and operators with explicitly supported
    semantics are folded. If evaluation would raise, the original expression
    shape is retained and left for the backend to report.
    """
    if not isinstance(expr, Expr):
        raise TypeError(f"optimize_expr expects an Expr, got {type(expr).__name__}.")

    if isinstance(expr, BinaryExpr):
        left = optimize_expr(expr.left)
        right = optimize_expr(expr.right)
        if (
            isinstance(left, Literal)
            and isinstance(right, Literal)
            and _safe_literal(left.value)
            and _safe_literal(right.value)
        ):
            operation = _FOLD_OPERATORS.get(expr.op)
            if operation is not None:
                try:
                    value = operation(left.value, right.value)
                except (ArithmeticError, TypeError, ValueError, OverflowError):
                    pass
                else:
                    if _safe_literal(value):
                        return Literal(value)
        return BinaryExpr(expr.op, left, right)

    if isinstance(expr, UnaryExpr):
        operand = optimize_expr(expr.operand)
        if isinstance(operand, Literal) and _safe_literal(operand.value):
            operation = operator.invert if expr.op == "~" else None
            if operation is not None:
                try:
                    value = operation(operand.value)
                except (ArithmeticError, TypeError, ValueError, OverflowError):
                    pass
                else:
                    if _safe_literal(value):
                        return Literal(value)
        return UnaryExpr(expr.op, operand)

    if isinstance(expr, WhenExpr):
        condition = optimize_expr(expr.condition)
        then_value = optimize_expr(expr.then_value)
        else_value = (
            optimize_expr(expr.else_value) if expr.else_value is not None else None
        )
        if isinstance(condition, Literal) and type(condition.value) is bool:
            selected = then_value if condition.value else else_value
            if selected is not None:
                return selected
        return WhenExpr(condition, then_value, else_value)

    if isinstance(expr, AccessorExpr):
        column = optimize_expr(expr.column)
        if isinstance(expr, StringMethodExpr):
            return StringMethodExpr(
                expr.method_name, column, dict(expr.kwargs), args=expr.args
            )
        if isinstance(expr, DatePartExpr):
            return DatePartExpr(
                expr.method_name,
                column,
                dict(expr.kwargs),
                is_property=expr.is_property,
            )
        return type(expr)(
            expr.namespace,
            expr.method_name,
            column,
            args=expr.args,
            kwargs=expr.kwargs,
            is_property=expr.is_property,
        )

    if isinstance(expr, (AggExpr, WindowExpr, MathExpr)):
        return type(expr)(
            expr.func, optimize_expr(expr.column), dict(expr.kwargs)
        )

    if isinstance(expr, (ColumnExpr, Literal)):
        return expr

    return expr


def _column_references(expr: Expr) -> set[str]:
    if isinstance(expr, ColumnExpr):
        return {expr.name}
    if isinstance(expr, BinaryExpr):
        return _column_references(expr.left) | _column_references(expr.right)
    if isinstance(expr, UnaryExpr):
        return _column_references(expr.operand)
    if isinstance(expr, WhenExpr):
        refs = _column_references(expr.condition) | _column_references(expr.then_value)
        if expr.else_value is not None:
            refs |= _column_references(expr.else_value)
        return refs
    if isinstance(expr, AccessorExpr):
        return _column_references(expr.column)
    if isinstance(expr, (AggExpr, WindowExpr, MathExpr)):
        return _column_references(expr.column)
    return set()


def _row_local(expr: Any) -> bool:
    if not isinstance(expr, Expr):
        return type(expr) in (bool, int, float, str, bytes, type(None))
    if isinstance(expr, (ColumnExpr, Literal)):
        return True
    if isinstance(expr, BinaryExpr):
        return _row_local(expr.left) and _row_local(expr.right)
    if isinstance(expr, UnaryExpr):
        return _row_local(expr.operand)
    if isinstance(expr, WhenExpr):
        return (
            _row_local(expr.condition)
            and _row_local(expr.then_value)
            and (expr.else_value is None or _row_local(expr.else_value))
        )
    if isinstance(expr, AccessorExpr):
        return _row_local(expr.column)
    if isinstance(expr, MathExpr):
        return _row_local(expr.column)
    return False


def optimize_plan(plan: list[tuple[str, tuple, dict]]) -> list[tuple[str, tuple, dict]]:
    """Fold expressions and safely push filters before independent mutations."""
    optimized = []
    for operation, args, kwargs in plan:
        if operation == "filter" and args and isinstance(args[0], Expr):
            args = (optimize_expr(args[0]),) + args[1:]
        elif operation == "mutate":
            kwargs = {
                name: optimize_expr(value) if isinstance(value, Expr) else value
                for name, value in kwargs.items()
            }
        optimized.append((operation, args, kwargs))

    index = 1
    while index < len(optimized):
        operation, args, _ = optimized[index]
        previous_op, _, previous_kwargs = optimized[index - 1]
        if operation == "filter" and previous_op == "mutate":
            filter_expr = args[0]
            targets = set(previous_kwargs)
            mutations_are_local = all(_row_local(value) for value in previous_kwargs.values())
            if (
                isinstance(filter_expr, Expr)
                and mutations_are_local
                and not (targets & _column_references(filter_expr))
            ):
                optimized[index - 1], optimized[index] = (
                    optimized[index],
                    optimized[index - 1],
                )
                index = max(1, index - 1)
                continue
        index += 1
    return optimized
