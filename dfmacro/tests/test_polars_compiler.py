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
        return df[price > 100]

    out = result()
    # Polars filter returns a DataFrame
    assert isinstance(out.raw, pl.DataFrame)
    assert out.raw.height == 2
    assert out.raw["price"].to_list() == [150, 200]


def test_polars_multi_condition_filter(df):
    @enable(df)
    def result():
        return df[(price > 100) & (category == "electronics")]

    out = result()
    assert out.raw.height == 2
    assert out.raw["price"].to_list() == [150, 200]


def test_polars_select(df):
    @enable(df)
    def result():
        return df.select(price, category)

    out = result()
    assert out.raw.columns == ["price", "category"]
    assert out.raw.height == 4


def test_polars_groupby_agg(df):
    @enable(df)
    def result():
        return df.groupby(category).agg(avg_price=price.mean())

    out = result()
    assert "avg_price" in out.raw.columns
    assert out.raw.height == 2


def test_polars_missing_column_raises(df):
    @enable(df)
    def result():
        return df[nonexistent > 1]

    with pytest.raises(CompileError, match="not found"):
        result()


def test_polars_where():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"price": [50, 150, 200, 80]})

    @enable(df)
    def result(df):
        return df.where(price > 100, 0)  # noqa: F821

    out = result(df)
    assert out.raw["price"].to_list() == [0, 150, 200, 0]


def test_polars_nlargest():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"price": [50, 150, 200, 80]})

    @enable(df)
    def result(df):
        return df.nlargest(2, price)  # noqa: F821

    out = result(df)
    assert out.raw["price"].to_list() == [200, 150]
