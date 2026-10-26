"""Tests for the optional Dask DataFrame backend."""

from __future__ import annotations

import pandas as pd
import pytest

dd = pytest.importorskip("dask.dataframe")

from dfmacro import DF, enable


@pytest.fixture
def ddf():
    frame = pd.DataFrame(
        {
            "price": [50, 150, 200, 80],
            "category": ["books", "electronics", "electronics", "books"],
        }
    )
    return dd.from_pandas(frame, npartitions=2)


def test_dask_expression_filter_is_lazy(ddf):
    @enable(ddf)
    def result(df):
        return df[price > 100]  # noqa: F821

    out = result(ddf)
    assert isinstance(out.raw, dd.DataFrame)
    assert out.raw.compute()["price"].tolist() == [150, 200]


def test_dask_filter_supports_string_accessor(ddf):
    @enable(ddf)
    def result(df):
        return df[category.str.contains("book")]  # noqa: F821

    out = result(ddf)
    assert out.raw.compute()["category"].tolist() == ["books", "books"]


def test_dask_select_and_mutate_chain(ddf):
    @enable(ddf)
    def result(df):
        return df.mutate(double=price * 2).select(category, double)  # noqa: F821

    out = result(ddf).raw.compute()
    assert out.columns.tolist() == ["category", "double"]
    assert out["double"].tolist() == [100, 300, 400, 160]


def test_dask_partition_expression_matches_pandas(ddf):
    @enable(ddf)
    def result(df):
        return df[(price > 75) & (price < 200)]  # noqa: F821

    actual = result(ddf).raw.compute().sort_values("price")
    expected = ddf.compute().query("75 < price < 200").sort_values("price")
    pd.testing.assert_frame_equal(actual, expected)


def test_dask_sort_uses_dask_dataframe_sort_values(ddf):
    @enable(ddf)
    def result(df):
        return df.order_by(price, descending=True)  # noqa: F821

    out = result(ddf).raw.compute()
    assert out["price"].tolist() == [200, 150, 80, 50]


def test_dask_groupby_and_join_report_unsupported_operations(ddf):
    with pytest.raises(NotImplementedError, match="Groupby aggregation.*Dask"):
        DF(ddf).groupby("category")

    with pytest.raises(NotImplementedError, match="Join.*Dask"):
        DF(ddf).join(ddf, on="category")
