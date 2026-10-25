# pyright: reportUndefinedVariable=false
"""Tests for the @enable decorator and the end-to-end flow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfmacro import enable, when
from dfmacro.compiler import CompileError


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "price": [50, 150, 200, 80],
            "category": ["books", "electronics", "electronics", "books"],
            "brand": ["a", "b", "b", "a"],
            "date": pd.to_datetime(
                ["2024-01-15", "2023-05-20", "2024-02-10", "2023-11-01"]
            ),
        }
    )


def test_basic_filter(df):
    @enable(df)
    def result():
        return df[price > 100]

    out = result()
    assert len(out) == 2
    assert list(out.raw["price"]) == [150, 200]


def test_multi_condition_filter(df):
    @enable(df)
    def result():
        return df[(price > 100) & (category == "electronics")]

    out = result()
    assert len(out) == 2


def test_string_key_still_works(df):
    @enable(df)
    def result():
        return df["price"]

    out = result()
    assert out.equals(df["price"])


def test_attribute_delegation(df):
    @enable(df)
    def result():
        return df.columns, df.shape

    cols, shape = result()
    assert "price" in cols
    assert shape == (4, 4)


def test_globals_cleaned_after_return(df):
    @enable(df)
    def result():
        return df[price > 100]

    result()
    g = globals()
    assert "price" not in g
    assert "category" not in g


def test_globals_cleaned_after_exception(df):
    @enable(df)
    def result():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        result()
    assert "price" not in globals()


def test_nested_function_does_not_see_injection(df):
    @enable(df)
    def outer():
        def inner():
            return price  # noqa: F821

        return inner

    inner = outer()
    with pytest.raises(NameError):
        inner()


def test_filter_with_arithmetic(df):
    @enable(df)
    def result():
        return df[price * 2 > 200]

    out = result()
    assert list(out.raw["price"]) == [150, 200]


def test_invert_filter(df):
    @enable(df)
    def result():
        return df[~(price > 100)]

    out = result()
    assert list(out.raw["price"]) == [50, 80]


def test_or_filter(df):
    @enable(df)
    def result():
        return df[(price > 150) | (brand == "a")]

    out = result()
    assert list(out.raw["price"]) == [50, 200, 80]


def test_missing_column_raises_compile_error(df):
    @enable(df)
    def result():
        return df[nonexistent > 1]

    with pytest.raises(CompileError, match="not found"):
        result()


def test_select_syntax(df):
    @enable(df)
    def result():
        return df.select(price, category)

    out = result()
    assert list(out.columns) == ["price", "category"]
    assert len(out) == 4


def test_groupby_agg_syntax(df):
    @enable(df)
    def result():
        return df.groupby(category).agg(avg_price=price.mean())

    out = result()
    assert "avg_price" in out.columns
    assert len(out) == 2


def test_chaining_filter_and_select(df):
    @enable(df)
    def result():
        return df[price > 100].select(price, brand)

    out = result()
    assert list(out.columns) == ["price", "brand"]
    assert len(out) == 2


def test_string_method_contains(df):
    @enable(df)
    def result():
        return df[category.str.contains("elect")]

    out = result()
    assert len(out) == 2


def test_mutate_new_column(df):
    @enable(df)
    def result():
        return df.mutate(double_price=price * 2)

    out = result()
    assert "double_price" in out.columns
    assert list(out.raw["double_price"]) == [100, 300, 400, 160]


def test_order_by(df):
    @enable(df)
    def result():
        return df.order_by(price, descending=True)

    out = result()
    assert list(out.raw["price"]) == [200, 150, 80, 50]


def test_when_otherwise(df):
    @enable(df)
    def result():
        return df.mutate(level=when(price > 100, "high").otherwise("low"))

    out = result()
    assert "level" in out.columns
    assert list(out.raw["level"]) == ["low", "high", "high", "low"]


def test_join_dataframes(df):
    other = pd.DataFrame(
        {"category": ["books", "electronics"], "tax_rate": [0.05, 0.15]}
    )

    @enable(df)
    def result():
        return df.join(other, on=category)

    out = result()
    assert "tax_rate" in out.columns
    assert len(out) == 4


def test_head_tail_slice(df):
    @enable(df)
    def result():
        return df.head(2)

    out = result()
    assert len(out) == 2
    assert list(out.raw["price"]) == [50, 150]


def test_drop_rename(df):
    @enable(df)
    def result():
        return df.drop(brand).rename(price=price_rmb)

    out = result()
    assert "brand" not in out.columns
    assert "price_rmb" in out.columns


def test_date_accessor(df):
    @enable(df)
    def result():
        return df[date.dt.year == 2024]

    out = result()
    assert len(out) == 2
    assert all(out.raw["date"].dt.year == 2024)


def test_explain(df):
    @enable(df)
    def result():
        return df[price > 100]

    out = result()
    # explain just returns repr of underlying df
    assert "price" in out.explain()


def test_mutate_chain_filter(df):
    @enable(df)
    def result():
        return df.mutate(double=price * 2)[double > 100]  # noqa: F821

    out = result()
    # price: 150 -> double 300 (keep)
    # price: 200 -> double 400 (keep)
    # price: 80  -> double 160 (keep)
    assert len(out) == 3
    assert list(out.raw["price"]) == [150, 200, 80]
    assert "double" in out.columns
