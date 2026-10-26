"""Tests for the DuckDB backend."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
duckdb = pytest.importorskip("duckdb")

from dfmacro import enable


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "price": [50, 150, 200, 80],
            "category": ["books", "electronics", "electronics", "books"],
        }
    )


def test_duckdb_basic_filter(df):
    @enable(df)
    def result():
        return df.to_duckdb()[price > 100]

    out = result()
    assert len(out.collect()) == 2
    assert list(out.collect()["price"]) == [150, 200]


def test_duckdb_complex_filter(df):
    @enable(df)
    def result():
        return df.to_duckdb()[(price > 100) & (category == "electronics")]

    out = result()
    assert len(out.collect()) == 2


def test_duckdb_groupby(df):
    @enable(df)
    def result():
        return (
            df.to_duckdb().groupby(category).agg(avg_price=price.mean())
        )

    out = result()
    assert "avg_price" in out.collect().columns
    assert len(out.collect()) == 2


def test_duckdb_mutate(df):
    @enable(df)
    def result():
        return df.to_duckdb().mutate(double=price * 2)

    out = result()
    assert "double" in out.collect().columns
    assert list(out.collect()["double"]) == [100, 300, 400, 160]


def test_duckdb_join(df):
    import pandas as pd

    other = pd.DataFrame(
        {"category": ["books", "electronics"], "tax_rate": [0.05, 0.15]}
    )

    @enable(df)
    def result():
        return df.to_duckdb().join(other, on=category)

    out = result()
    assert "tax_rate" in out.collect().columns
    assert len(out.collect()) == 4


def test_duckdb_lazy_pipeline(df):
    @enable(df)
    def result():
        return (
            df.to_duckdb().lazy()[price > 100].mutate(double_price=price * 2).collect()
        )

    out = result()
    assert len(out.collect()) == 2
    assert "double_price" in out.collect().columns
    assert list(out.collect()["double_price"]) == [300, 400]
