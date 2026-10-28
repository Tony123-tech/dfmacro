"""Tests for lazy plan construction and execution."""

import pandas as pd
import pytest

from dfmacro import DF
from dfmacro.compiler import CompileError
from dfmacro.expr import ColumnExpr


@pytest.fixture
def products():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "product": ["book", "phone", "lamp"],
            "price": [12, 300, 45],
        }
    )


@pytest.fixture
def inventory():
    return pd.DataFrame(
        {
            "id": [1, 2, 4],
            "stock": [10, 0, 7],
        }
    )


def test_lazy_join_collects_with_pandas_merge(products, inventory):
    left = DF(products).lazy()
    right = DF(inventory).lazy()

    plan = left.join(right, on="id", how="inner")
    result = plan.collect().raw

    assert result[["id", "product", "price", "stock"]].to_dict("list") == {
        "id": [1, 2],
        "product": ["book", "phone"],
        "price": [12, 300],
        "stock": [10, 0],
    }


def test_lazy_join_preserves_branch_plans_and_filters_after_join(products, inventory):
    left = DF(products).lazy().filter(ColumnExpr("price") > 20)
    right = DF(inventory).lazy().filter(ColumnExpr("stock") >= 0)
    plan = left.join(right, on="id").filter(ColumnExpr("stock") > 0)

    assert plan._plan[-1][0] == "filter"
    join_node = plan._plan[-2]
    assert join_node[0] == "join"
    assert join_node[1][0] is right
    result = plan.collect().raw

    assert result[["id", "product", "stock"]].to_dict("list") == {
        "id": [],
        "product": [],
        "stock": [],
    }


def test_lazy_join_supports_multiple_keys():
    left = pd.DataFrame({"region": ["east", "west"], "id": [1, 1], "left": [10, 20]})
    right = pd.DataFrame(
        {"region": ["east", "west"], "id": [1, 2], "right": [100, 200]}
    )

    result = DF(left).lazy().join(DF(right).lazy(), on=["region", "id"]).collect().raw

    assert result[["region", "id", "left", "right"]].to_dict("list") == {
        "region": ["east"],
        "id": [1],
        "left": [10],
        "right": [100],
    }


def test_lazy_join_rejects_non_lazy_branch(products):
    with pytest.raises(TypeError, match="another LazyDF"):
        DF(products).lazy().join(DF(products))


def test_lazy_join_validates_join_type(products, inventory):
    with pytest.raises(ValueError, match="how must be"):
        DF(products).lazy().join(DF(inventory).lazy(), how="invalid")


def test_lazy_join_reports_missing_key(products):
    right = pd.DataFrame({"other_id": [1]})

    with pytest.raises(CompileError, match="missing"):
        DF(products).lazy().join(DF(right).lazy(), on="id").collect()


def test_lazy_explain_shows_operations_and_rename_lineage(products, inventory):
    description = (
        DF(products)
        .lazy()
        .filter(ColumnExpr("price") > 20)
        .join(DF(inventory).lazy(), on="id")
        .rename(product="item")
        .explain()
    )

    assert "Filter" in description
    assert "Join" in description
    assert "Rename (product -> item)" in description
    assert "item" in description


def test_lazy_rename_collects_with_pandas(products):
    result = DF(products).lazy().rename(product="item").collect().raw
    assert "item" in result.columns
    assert "product" not in result.columns


def test_polars_lazy_plan_uses_native_lazy_collection():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

    result = (
        DF(frame)
        .lazy()
        .filter(ColumnExpr("value") > 10)
        .mutate(double=ColumnExpr("value") * 2)
        .select(ColumnExpr("id"), ColumnExpr("double"))
        .collect()
        .raw
    )

    assert isinstance(result, pl.DataFrame)
    assert result.to_dict(as_series=False) == {"id": [2, 3], "double": [40, 60]}


def test_polars_lazy_join_uses_native_join():
    pl = pytest.importorskip("polars")
    left = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    right = pl.DataFrame({"id": [1, 2], "stock": [4, 5]})

    result = (
        DF(left)
        .lazy()
        .join(DF(right).lazy(), on="id")
        .collect()
        .raw.sort("id")
    )
    assert result.to_dict(as_series=False) == {
        "id": [1, 2],
        "name": ["a", "b"],
        "stock": [4, 5],
    }


def test_dask_lazy_join_uses_native_merge():
    dd = pytest.importorskip("dask.dataframe")
    left = dd.from_pandas(
        pd.DataFrame({"id": [1, 2], "value": [10, 20]}), npartitions=2
    )
    right = dd.from_pandas(
        pd.DataFrame({"id": [1, 2], "stock": [4, 5]}), npartitions=2
    )

    result = DF(left).lazy().join(DF(right).lazy(), on="id").collect().raw.compute()
    assert result.sort_values("id")["stock"].tolist() == [4, 5]


def test_dask_lazy_groupby_uses_native_aggregation():
    dd = pytest.importorskip("dask.dataframe")
    frame = dd.from_pandas(
        pd.DataFrame({"group": ["a", "a", "b"], "value": [2, 4, 10]}),
        npartitions=2,
    )

    result = (
        DF(frame)
        .lazy()
        .groupby("group")
        .agg(average=ColumnExpr("value").mean())
        .collect()
        .raw.compute()
        .sort_values("group")
        .reset_index(drop=True)
    )
    assert result.to_dict("list") == {"group": ["a", "b"], "average": [3.0, 10.0]}
