"""Tests for the Polars backend compiler and DF wrapper."""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from dfmacro import enable
from dfmacro.compiler import CompileError


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "price": [50, 150, 200, 80],
            "category": ["books", "electronics", "electronics", "books"],
            "brand": ["a", "b", "b", "a"],
        }
    )


def test_polars_basic_filter(df):
    @enable(df)
    def result():
        return df[price > 100]  # noqa: F821

    out = result()
    # Polars filter returns a DataFrame
    assert isinstance(out.raw, pl.DataFrame)
    assert out.raw.height == 2
    assert out.raw["price"].to_list() == [150, 200]


def test_polars_multi_condition_filter(df):
    @enable(df)
    def result():
        return df[(price > 100) & (category == "electronics")]  # noqa: F821

    out = result()
    assert out.raw.height == 2
    assert out.raw["price"].to_list() == [150, 200]


def test_polars_select(df):
    @enable(df)
    def result():
        return df.select(price, category)  # noqa: F821

    out = result()
    assert out.raw.columns == ["price", "category"]
    assert out.raw.height == 4


def test_polars_groupby_agg(df):
    @enable(df)
    def result():
        return df.groupby(category).agg(avg_price=price.mean())  # noqa: F821

    out = result()
    assert "avg_price" in out.raw.columns
    assert out.raw.height == 2


def test_polars_missing_column_raises(df):
    @enable(df)
    def result():
        return df[nonexistent > 1]  # noqa: F821

    with pytest.raises(CompileError, match="not found"):
        result()
