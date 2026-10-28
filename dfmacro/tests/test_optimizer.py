"""Tests for conservative expression and plan optimization."""

import pandas as pd

from dfmacro import DF
from dfmacro.expr import BinaryExpr, ColumnExpr, Literal, structural_eq
from dfmacro.optimizer import optimize_expr, optimize_plan


def test_optimizer_folds_literal_arithmetic():
    optimized = optimize_expr(Literal(1) + Literal(2) * Literal(3))

    assert structural_eq(optimized, Literal(7))


def test_optimizer_keeps_invalid_constant_operation_unfolded():
    expression = Literal(1) / Literal(0)

    assert isinstance(optimize_expr(expression), BinaryExpr)


def test_plan_pushes_filter_before_independent_row_local_mutation():
    predicate = ColumnExpr("price") > 10
    plan = [
        ("mutate", (), {"twice": ColumnExpr("price") * 2}),
        ("filter", (predicate,), {}),
    ]

    optimized = optimize_plan(plan)

    assert [operation for operation, _, _ in optimized] == ["filter", "mutate"]


def test_plan_does_not_push_filter_across_replacement_of_referenced_column():
    predicate = ColumnExpr("price") > 10
    plan = [
        ("mutate", (), {"price": ColumnExpr("price") * 2}),
        ("filter", (predicate,), {}),
    ]

    optimized = optimize_plan(plan)

    assert [operation for operation, _, _ in optimized] == ["mutate", "filter"]


def test_plan_does_not_push_filter_across_window_mutation():
    predicate = ColumnExpr("price") > 10
    plan = [
        ("mutate", (), {"running": ColumnExpr("price").cumsum()}),
        ("filter", (predicate,), {}),
    ]

    optimized = optimize_plan(plan)

    assert [operation for operation, _, _ in optimized] == ["mutate", "filter"]


def test_plan_does_not_push_filter_across_callable_mutation():
    predicate = ColumnExpr("price") > 10
    plan = [
        ("mutate", (), {"position": lambda frame: range(len(frame))}),
        ("filter", (predicate,), {}),
    ]

    optimized = optimize_plan(plan)

    assert [operation for operation, _, _ in optimized] == ["mutate", "filter"]


def test_filter_and_head_pipeline_collects_optimized_plan():
    frame = pd.DataFrame({"price": [4, 20, 12, 30], "category": ["a", "b", "c", "d"]})

    result = DF(frame).lazy()[ColumnExpr("price") > 10].head(1).collect()

    assert result.raw["price"].tolist() == [20]
