"""Tests for compiling LazyDF pipelines to SQL."""

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from dfmacro import DF, compile_to_sql
from dfmacro.expr import ColumnExpr
from dfmacro.compiler import CompileError


def test_sql_pipeline_compiles_filter_select_order_and_head():
    frame = pd.DataFrame({"price": [50, 150, 200, 80], "category": ["a", "b", "c", "d"]})
    plan = (
        DF(frame)
        .lazy()[ColumnExpr("price") > 70]
        .select(ColumnExpr("category"), ColumnExpr("price"))
        .order_by(ColumnExpr("price"))
        .head(2)
    )

    sql = compile_to_sql(plan)
    connection = duckdb.connect()
    connection.register("df", frame)
    result = connection.execute(sql).fetchdf()

    assert result.to_dict("list") == {"category": ["d", "b"], "price": [80, 150]}


def test_sql_pipeline_compiles_groupby_aggregation():
    frame = pd.DataFrame({"category": ["a", "a", "b"], "price": [10, 20, 30]})
    plan = DF(frame).lazy().groupby(ColumnExpr("category")).agg(
        average=ColumnExpr("price").mean()
    )

    sql = compile_to_sql(plan, table_name="analytics.products")
    connection = duckdb.connect()
    connection.execute("CREATE SCHEMA analytics")
    connection.register("_products", frame)
    connection.execute(
        "CREATE TABLE analytics.products AS SELECT * FROM _products"
    )

    result = connection.execute(sql).fetchdf()

    assert dict(zip(result["category"], result["average"])) == {"a": 15.0, "b": 30.0}


def test_sql_tail_requires_explicit_order():
    frame = pd.DataFrame({"price": [1, 2, 3]})
    plan = DF(frame).lazy().tail(1)

    with pytest.raises(CompileError, match="requires a preceding order_by"):
        compile_to_sql(plan)


def test_sql_tail_preserves_requested_order():
    frame = pd.DataFrame({"price": [1, 3, 2]})
    plan = DF(frame).lazy().order_by(ColumnExpr("price")).tail(2)

    sql = compile_to_sql(plan)
    connection = duckdb.connect()
    connection.register("df", frame)

    assert connection.execute(sql).fetchdf()["price"].tolist() == [2, 3]


def test_sql_pipeline_supports_literal_mutations():
    frame = pd.DataFrame({"price": [1, 2]})
    plan = DF(frame).lazy().mutate(label="fixed")

    sql = compile_to_sql(plan)
    connection = duckdb.connect()
    connection.register("df", frame)

    assert connection.execute(sql).fetchdf()["label"].tolist() == ["fixed", "fixed"]


def test_sql_pipeline_compiles_two_table_join_and_executes():
    left = pd.DataFrame({"id": [1, 2, 3], "product": ["book", "phone", "lamp"]})
    right = pd.DataFrame({"id": [1, 2, 4], "stock": [10, 0, 7]})
    plan = DF(left).lazy().join(DF(right).lazy(), on="id").filter(
        ColumnExpr("stock") > 0
    )

    sql = compile_to_sql(plan)
    assert '"df1".*' in sql
    assert '"df2".* EXCLUDE ("id")' in sql
    assert '"df1"."id" = "df2"."id"' in sql
    connection = duckdb.connect()
    connection.register("df1", left)
    connection.register("df2", right)
    result = connection.execute(sql).fetchdf()

    assert result[["id", "product", "stock"]].to_dict("list") == {
        "id": [1],
        "product": ["book"],
        "stock": [10],
    }


def test_sql_pipeline_compiles_join_of_pre_filtered_lazy_branches():
    left = pd.DataFrame({"id": [1, 2, 3], "amount": [10, 30, 50]})
    right = pd.DataFrame({"id": [1, 2, 3], "status": ["ok", "skip", "ok"]})
    plan = (
        DF(left)
        .lazy()
        .filter(ColumnExpr("amount") >= 30)
        .join(
            DF(right).lazy().filter(ColumnExpr("status") == "ok"),
            on="id",
            how="left",
        )
    )

    sql = compile_to_sql(plan)
    connection = duckdb.connect()
    connection.register("df1", left)
    connection.register("df2", right)
    result = connection.execute(sql).fetchdf()

    result = result.sort_values("id").reset_index(drop=True)
    assert result[["id", "amount"]].to_dict("list") == {
        "id": [2, 3],
        "amount": [30, 50],
    }
    assert pd.isna(result.loc[0, "status"])
    assert result.loc[1, "status"] == "ok"


def test_sql_join_uses_pandas_suffixes_for_colliding_columns():
    left = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    right = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
    plan = DF(left).lazy().join(DF(right).lazy(), on="id").filter(
        ColumnExpr("value_y") > 100
    )

    connection = duckdb.connect()
    connection.register("df1", left)
    connection.register("df2", right)
    result = connection.execute(compile_to_sql(plan)).fetchdf()

    assert result.to_dict("list") == {"id": [2], "value_x": [20], "value_y": [200]}


def test_sql_full_join_preserves_keys_from_both_sides():
    left = pd.DataFrame({"id": [1, 2], "left_value": ["a", "b"]})
    right = pd.DataFrame({"id": [2, 3], "right_value": ["x", "y"]})
    plan = DF(left).lazy().join(DF(right).lazy(), on="id", how="outer")

    connection = duckdb.connect()
    connection.register("df1", left)
    connection.register("df2", right)
    result = connection.execute(compile_to_sql(plan)).fetchdf().sort_values("id")

    assert result["id"].tolist() == [1, 2, 3]
    assert result["left_value"].tolist()[:2] == ["a", "b"]
    assert pd.isna(result.iloc[2]["left_value"])
    assert result.iloc[2]["right_value"] == "y"


def test_sql_pipeline_compiles_a_three_table_join_with_distinct_aliases():
    customers = pd.DataFrame({"id": [1, 2], "name": ["Ada", "Grace"]})
    regions = pd.DataFrame({"id": [1, 2], "region": ["east", "west"]})
    scores = pd.DataFrame({"id": [1, 2], "score": [90, 95]})
    plan = (
        DF(customers)
        .lazy()
        .join(DF(regions).lazy(), on="id")
        .join(DF(scores).lazy(), on="id")
        .filter(ColumnExpr("score") > 90)
    )

    connection = duckdb.connect()
    connection.register("df1", customers)
    connection.register("df2", regions)
    connection.register("df3", scores)
    result = connection.execute(compile_to_sql(plan)).fetchdf()

    assert result.to_dict("list") == {
        "id": [2],
        "name": ["Grace"],
        "region": ["west"],
        "score": [95],
    }
