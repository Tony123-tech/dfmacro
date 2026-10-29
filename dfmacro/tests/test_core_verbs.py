"""Tests for core DataFrame verbs on the DF wrapper."""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from dfmacro import DF, df_wrapper, enable
from dfmacro.expr import ColumnExpr


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "price": [50, 150, 200, 80],
            "category": ["books", "electronics", "electronics", "books"],
        }
    )


@pytest.fixture
def numeric_df():
    return pd.DataFrame({"price": [50, 150, 200, 80]})


def test_where_keeps_matching(df):
    @enable(df)
    def result(df):
        return df.where(price > 100, 0)  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [0, 150, 200, 0]


def test_expression_filter_is_cached(df, monkeypatch):
    wrapper = DF(df)
    expression = ColumnExpr("price") > 100
    compile_calls = 0
    original_compile_expr = df_wrapper.compile_expr

    def counting_compile_expr(expr, frame):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile_expr(expr, frame)

    monkeypatch.setattr(df_wrapper, "compile_expr", counting_compile_expr)

    first = wrapper[expression]
    second = wrapper[ColumnExpr("price") > 100]

    assert first is second
    assert compile_calls == 1
    assert len(wrapper._cache) == 1


def test_unserializable_expression_bypasses_cache(df, monkeypatch):
    wrapper = DF(df)
    expression = ColumnExpr("price") > 100

    def fail_pickle(*args, **kwargs):
        raise TypeError("value cannot be serialized")

    monkeypatch.setattr(df_wrapper.pickle, "dumps", fail_pickle)

    first = wrapper[expression]
    second = wrapper[expression]

    assert first is not second
    assert first.raw.equals(second.raw)
    assert wrapper._cache == {}


@pytest.mark.parametrize(
    ("value", "expected_prices"),
    [(True, [50, 150, 200, 80]), (False, [])],
)
def test_constant_predicates_filter_all_or_no_rows(df, value, expected_prices):
    from dfmacro.expr import Literal

    result = DF(df)[Literal(value)]

    assert result.raw["price"].tolist() == expected_prices


def test_mask_replaces_matching(df):
    @enable(df)
    def result(df):
        return df.mask(price > 100, 0)  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [50, 0, 0, 80]


def test_nlargest(df):
    @enable(df)
    def result(df):
        return df.nlargest(2, price)  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [200, 150]


def test_nsmallest(df):
    @enable(df)
    def result(df):
        return df.nsmallest(2, price)  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [50, 80]


def test_shift(df):
    @enable(df)
    def result(df):
        return df.shift(1)

    out = result(df)
    assert pd.isna(out.raw["price"].iloc[0])
    assert list(out.raw["price"].iloc[1:]) == [50, 150, 200]


def test_diff(numeric_df):
    @enable(numeric_df)
    def result(df):
        return df.diff()

    out = result(numeric_df)
    assert pd.isna(out.raw["price"].iloc[0])
    assert list(out.raw["price"].iloc[1:]) == [100, 50, -120]


def test_rank(numeric_df):
    @enable(numeric_df)
    def result(df):
        return df.rank()

    out = result(numeric_df)
    assert list(out.raw["price"]) == [1, 3, 4, 2]


def test_clip(numeric_df):
    @enable(numeric_df)
    def result(df):
        return df.clip(lower=60, upper=180)

    out = result(numeric_df)
    assert list(out.raw["price"]) == [60, 150, 180, 80]


def test_replace(df):
    @enable(df)
    def result(df):
        return df.replace("books", "media")

    out = result(df)
    assert list(out.raw["category"]) == [
        "media",
        "electronics",
        "electronics",
        "media",
    ]


def test_sort_values(df):
    @enable(df)
    def result(df):
        return df.sort_values(by=price, ascending=False)  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [200, 150, 80, 50]


def test_reset_index(df):
    df.index = [10, 11, 12, 13]

    @enable(df)
    def result(df):
        return df.reset_index()

    out = result(df)
    assert list(out.raw.columns) == ["index", "price", "category"]
    assert list(out.raw["index"]) == [10, 11, 12, 13]


def test_set_index(df):
    @enable(df)
    def result(df):
        return df.set_index(category)  # noqa: F821

    out = result(df)
    assert list(out.raw.index) == ["books", "electronics", "electronics", "books"]
    assert "category" not in out.raw.columns


def test_assign_expression(df):
    @enable(df)
    def result(df):
        return df.assign(double_price=price * 2)  # noqa: F821

    out = result(df)
    assert list(out.raw["double_price"]) == [100, 300, 400, 160]


def test_filter_expression(df):
    @enable(df)
    def result(df):
        return df.filter(price > 100)  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [150, 200]


def test_filter_labels(df):
    @enable(df)
    def result(df):
        return df.filter(items=["price"])

    out = result(df)
    assert list(out.raw.columns) == ["price"]


def test_isin(df):
    @enable(df)
    def result(df):
        return df.isin({"price": [50, 200], "category": ["books"]})

    out = result(df)
    assert list(out.raw["price"]) == [True, False, True, False]
    assert list(out.raw["category"]) == [True, False, False, True]


def test_between_filters_rows(df):
    @enable(df)
    def result(df):
        return df.between(price, 80, 150)  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [150, 80]


@pytest.mark.parametrize(
    ("inclusive", "expected"),
    [
        ("both", [150, 200, 80]),
        ("neither", [150]),
        ("left", [150, 80]),
        ("right", [150, 200]),
    ],
)
def test_between_inclusive(df, inclusive, expected):
    @enable(df)
    def result(df):
        return df.between(price, 80, 200, inclusive=inclusive)  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == expected


def test_polars_core_verbs():
    pl = pytest.importorskip("polars")
    source = pl.DataFrame(
        {"price": [50, 150, 200, 80], "category": ["a", "b", "b", "a"]}
    )

    @enable(source)
    def result(df):
        return (
            df.sort_values(by=price, ascending=False)  # noqa: F821
            .assign(double_price=price * 2)  # noqa: F821
            .filter(price > 100)  # noqa: F821
        )

    out = result(source)
    assert out.raw["price"].to_list() == [200, 150]
    assert out.raw["double_price"].to_list() == [400, 300]
    assert DF(source).isin({"price": [50, 200]}).raw["price"].to_list() == [
        True,
        False,
        True,
        False,
    ]

    @enable(source)
    def between_result(df):
        return df.between(price, 80, 150)  # noqa: F821

    assert between_result(source).raw["price"].to_list() == [150, 80]


def test_polars_reset_index_and_set_index_behavior():
    pl = pytest.importorskip("polars")
    source = pl.DataFrame({"price": [50, 150]})

    @enable(source)
    def reset(df):
        return df.reset_index()

    @enable(source)
    def reset_drop(df):
        return df.reset_index(drop=True)

    @enable(source)
    def set_index(df):
        return df.set_index(price)  # noqa: F821

    assert reset(source).raw.equals(source)
    assert reset_drop(source).raw.equals(source)
    with pytest.raises(NotImplementedError, match="pandas-style indexes"):
        set_index(source)


def test_describe_pandas_and_polars(numeric_df):
    pl = pytest.importorskip("polars")
    pandas_result = DF(numeric_df).describe().raw
    polars_result = DF(pl.DataFrame({"price": [50, 150, 200, 80]})).describe().raw

    assert "mean" in pandas_result.index
    assert pandas_result.loc["mean", "price"] == 120
    assert "statistic" in polars_result.columns
    assert polars_result.filter(pl.col("statistic") == "mean")["price"].item() == 120


def test_value_counts_pandas_and_polars(df):
    pl = pytest.importorskip("polars")
    pandas_result = DF(df).value_counts("category").raw
    polars_result = (
        DF(pl.DataFrame({"category": ["books", "electronics", "electronics", "books"]}))
        .value_counts("category")
        .raw
    )

    assert pandas_result.to_dict("records") == [
        {"category": "books", "count": 2},
        {"category": "electronics", "count": 2},
    ]
    polars_result = polars_result.sort("category")
    assert polars_result.columns == ["category", "count"]
    assert polars_result["category"].to_list() == ["books", "electronics"]
    assert polars_result["count"].to_list() == [2, 2]


def test_corr_pandas_and_polars():
    pl = pytest.importorskip("polars")
    source = pd.DataFrame({"x": [1, 2, 3], "y": [3, 5, 7], "label": ["a", "b", "c"]})
    pandas_result = DF(source).corr().raw
    polars_result = (
        DF(pl.DataFrame({"x": [1, 2, 3], "y": [3, 5, 7], "label": ["a", "b", "c"]}))
        .corr()
        .raw
    )

    assert pandas_result.loc["x", "y"] == pytest.approx(1)
    assert polars_result.columns == ["variable", "x", "y"]
    assert polars_result["variable"].to_list() == ["x", "y"]
    assert polars_result["y"].to_list() == pytest.approx([1, 1])


def test_cov_pandas_and_polars():
    pl = pytest.importorskip("polars")
    source = pd.DataFrame({"x": [1, 2, 3], "y": [3, 5, 7]})
    pandas_result = DF(source).cov().raw
    polars_result = DF(pl.DataFrame({"x": [1, 2, 3], "y": [3, 5, 7]})).cov().raw

    assert pandas_result.loc["x", "y"] == pytest.approx(2)
    assert polars_result.columns == ["variable", "x", "y"]
    assert polars_result["variable"].to_list() == ["x", "y"]
    assert polars_result["y"].to_list() == pytest.approx([2, 4])


def test_pct_change_pandas_and_polars(numeric_df):
    pl = pytest.importorskip("polars")
    pandas_result = DF(numeric_df).pct_change("price").raw["price"]
    polars_result = (
        DF(pl.DataFrame({"price": [50, 150, 200, 80]})).pct_change("price").raw["price"]
    )

    assert pandas_result.iloc[1:].tolist() == pytest.approx([2, 1 / 3, -0.6])
    assert polars_result[1:].to_list() == pytest.approx([2, 1 / 3, -0.6])


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("shift", [None, 50, 150, 200]),
        ("diff", [None, 100, 50, -120]),
        ("cumsum", [50, 200, 400, 480]),
        ("cummax", [50, 150, 200, 200]),
        ("cummin", [50, 50, 50, 50]),
    ],
)
def test_window_methods_pandas_and_polars(numeric_df, method, expected):
    pl = pytest.importorskip("polars")
    pandas_df = DF(numeric_df)
    polars_df = DF(pl.DataFrame({"price": [50, 150, 200, 80]}))

    pandas_result = getattr(pandas_df, method)("price").raw["price"].tolist()
    polars_result = getattr(polars_df, method)("price").raw["price"].to_list()

    if expected[0] is None:
        assert pd.isna(pandas_result[0])
        assert polars_result[0] is None
        expected_values = expected[1:]
        pandas_result = pandas_result[1:]
        polars_result = polars_result[1:]
    else:
        expected_values = expected
    assert pandas_result == expected_values
    assert polars_result == expected_values


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("shift", [None, None, 50, 150]),
        ("diff", [None, None, 150, -70]),
    ],
)
def test_window_methods_periods(numeric_df, method, expected):
    pl = pytest.importorskip("polars")
    pandas_result = getattr(DF(numeric_df), method)("price", periods=2).raw["price"]
    polars_result = getattr(DF(pl.DataFrame({"price": [50, 150, 200, 80]})), method)(
        "price", periods=2
    ).raw["price"]

    assert pd.isna(pandas_result.iloc[0])
    assert pd.isna(pandas_result.iloc[1])
    assert polars_result[:2].to_list() == [None, None]
    assert pandas_result.iloc[2:].tolist() == expected[2:]
    assert polars_result[2:].to_list() == expected[2:]


def test_melt_pandas_and_polars():
    pl = pytest.importorskip("polars")
    source = pd.DataFrame({"id": [1, 2], "left": [10, 20], "right": [30, 40]})
    pandas_result = DF(source).melt(id_vars="id", value_vars=["left", "right"]).raw
    polars_result = (
        DF(pl.DataFrame({"id": [1, 2], "left": [10, 20], "right": [30, 40]}))
        .melt(id_vars="id", value_vars=["left", "right"])
        .raw
    )

    assert pandas_result.to_dict("records") == [
        {"id": 1, "variable": "left", "value": 10},
        {"id": 2, "variable": "left", "value": 20},
        {"id": 1, "variable": "right", "value": 30},
        {"id": 2, "variable": "right", "value": 40},
    ]
    assert polars_result.columns == ["id", "variable", "value"]
    assert polars_result["variable"].to_list() == ["left", "left", "right", "right"]
    assert polars_result["value"].to_list() == [10, 20, 30, 40]


def test_pivot_pandas_and_polars():
    pl = pytest.importorskip("polars")
    data = {
        "date": ["2024-01", "2024-01", "2024-02", "2024-02"],
        "kind": ["a", "b", "a", "b"],
        "value": [1, 2, 3, 4],
    }
    pandas_result = DF(pd.DataFrame(data)).pivot("date", "kind", "value").raw
    polars_result = DF(pl.DataFrame(data)).pivot("date", "kind", "value").raw

    assert pandas_result.loc["2024-01", "a"] == 1
    assert pandas_result.loc["2024-02", "b"] == 4
    assert polars_result.sort("date")["a"].to_list() == [1, 3]
    assert polars_result.sort("date")["b"].to_list() == [2, 4]


def test_pivot_table_pandas_and_polars_aggregates_duplicates():
    pl = pytest.importorskip("polars")
    data = {
        "date": ["2024-01", "2024-01", "2024-01"],
        "kind": ["a", "a", "b"],
        "value": [1, 3, 8],
    }
    pandas_result = (
        DF(pd.DataFrame(data)).pivot_table("date", "kind", "value", aggfunc="sum").raw
    )
    polars_result = (
        DF(pl.DataFrame(data)).pivot_table("date", "kind", "value", aggfunc="sum").raw
    )

    assert pandas_result.loc["2024-01", "a"] == 4
    assert polars_result["a"].item() == 4
    assert polars_result["b"].item() == 8


@pytest.mark.parametrize("func", ["mean", "sum"])
def test_rolling_pandas_and_polars(func):
    pl = pytest.importorskip("polars")
    source = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    pandas_result = DF(source).rolling(2, "value", func=func).raw["value"]
    polars_result = (
        DF(pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}))
        .rolling(2, "value", func=func)
        .raw["value"]
    )

    expected = [1.5, 2.5, 3.5] if func == "mean" else [3.0, 5.0, 7.0]
    assert pandas_result.iloc[1:].tolist() == pytest.approx(expected)
    assert polars_result[1:].to_list() == pytest.approx(expected)


def test_rolling_reports_unsupported_function():
    with pytest.raises(ValueError, match="Unsupported rolling function 'not_a_stat'"):
        DF(pd.DataFrame({"value": [1.0, 2.0]})).rolling(2, "value", func="not_a_stat")


def test_ewm_mean_pandas_and_polars():
    pl = pytest.importorskip("polars")
    source = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    pandas_result = DF(source).ewm(2, "value").raw["value"].tolist()
    polars_result = (
        DF(pl.DataFrame({"value": [1.0, 2.0, 3.0]}))
        .ewm(2, "value")
        .raw["value"]
        .to_list()
    )

    assert polars_result == pytest.approx(pandas_result)


def test_resample_pandas_and_polars():
    pl = pytest.importorskip("polars")
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    pandas_source = pd.DataFrame({"value": [1, 2, 3, 4]}, index=dates)
    polars_source = pl.DataFrame(
        {
            "date": pl.datetime_range(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 4),
                interval="1d",
                eager=True,
            ),
            "value": [1, 2, 3, 4],
        }
    )
    pandas_result = DF(pandas_source).resample("2D", "value").raw
    polars_result = DF(polars_source).resample("2d", "value").raw

    assert pandas_result["value"].tolist() == [3, 7]
    assert polars_result["value"].to_list() == [3, 7]


def test_asfreq_pandas_only(numeric_df):
    dates = pd.date_range("2024-01-01", periods=3, freq="2D")
    source = pd.DataFrame({"value": [1, 3, 5]}, index=dates)
    result = DF(source).asfreq("D").raw
    assert result["value"].iloc[[0, 2, 4]].tolist() == [1, 3, 5]
    assert result["value"].iloc[[1, 3]].isna().all()

    pl = pytest.importorskip("polars")
    with pytest.raises(NotImplementedError, match="only supported for pandas"):
        DF(pl.DataFrame({"value": [1, 3, 5]})).asfreq("1d")


def test_interpolate_pandas_and_polars():
    pl = pytest.importorskip("polars")
    pandas_result = (
        DF(pd.DataFrame({"value": [1.0, float("nan"), 3.0]})).interpolate().raw
    )
    polars_result = DF(pl.DataFrame({"value": [1.0, None, 3.0]})).interpolate().raw

    assert pandas_result["value"].tolist() == [1.0, 2.0, 3.0]
    assert polars_result["value"].to_list() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_directional_fill_pandas_and_polars(method):
    pl = pytest.importorskip("polars")
    pandas_source = pd.DataFrame({"value": [1.0, float("nan"), 3.0]})
    polars_source = pl.DataFrame({"value": [1.0, None, 3.0]})

    pandas_result = getattr(DF(pandas_source), method)().raw["value"].tolist()
    polars_result = getattr(DF(polars_source), method)().raw["value"].to_list()
    expected = [1.0, 1.0, 3.0] if method == "ffill" else [1.0, 3.0, 3.0]

    assert pandas_result == expected
    assert polars_result == expected


def test_nunique_pandas_and_polars(df):
    pl = pytest.importorskip("polars")
    pandas_result = DF(df).nunique()
    polars_result = (
        DF(
            pl.DataFrame(
                {
                    "price": [50, 150, 200, 80],
                    "category": ["books", "electronics", "electronics", "books"],
                }
            )
        )
        .nunique()
        .raw
    )

    assert pandas_result.to_dict() == {"price": 4, "category": 2}
    assert polars_result.row(0, named=True) == {"price": 4, "category": 2}


def test_mode_pandas_and_polars():
    pl = pytest.importorskip("polars")
    pandas_result = DF(pd.DataFrame({"value": [1, 1, 2, 3]})).mode().raw
    polars_result = DF(pl.DataFrame({"value": [1, 1, 2, 3]})).mode().raw

    assert pandas_result["value"].tolist() == [1]
    assert polars_result["value"].to_list() == [1]


def test_skew_and_kurt_pandas_and_polars():
    pl = pytest.importorskip("polars")
    values = [1.0, 2.0, 4.0, 8.0, 16.0]
    pandas_df = pd.DataFrame({"value": values})
    polars_df = pl.DataFrame({"value": values})

    assert DF(polars_df).skew().raw["value"].item() == pytest.approx(
        DF(pandas_df).skew()["value"]
    )
    assert DF(polars_df).kurt().raw["value"].item() == pytest.approx(
        DF(pandas_df).kurt()["value"]
    )


def test_merge_pandas_and_polars():
    pl = pytest.importorskip("polars")
    left = pd.DataFrame({"key": [1, 2], "left": ["a", "b"]})
    right = pd.DataFrame({"key": [2, 3], "right": ["x", "y"]})
    pandas_result = DF(left).merge(right, on="key").raw

    polars_result = (
        DF(pl.DataFrame({"key": [1, 2], "left": ["a", "b"]}))
        .merge(pl.DataFrame({"key": [2, 3], "right": ["x", "y"]}), on="key")
        .raw
    )
    assert pandas_result.to_dict("records") == [{"key": 2, "left": "b", "right": "x"}]
    assert polars_result.to_dicts() == [{"key": 2, "left": "b", "right": "x"}]


@pytest.mark.parametrize("axis", [0, 1])
def test_concat_pandas_and_polars(axis):
    pl = pytest.importorskip("polars")
    if axis == 0:
        pandas_frames = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
        polars_frames = [pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [2]})]
    else:
        pandas_frames = [pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]})]
        polars_frames = [pl.DataFrame({"a": [1]}), pl.DataFrame({"b": [2]})]

    pandas_result = DF.concat(pandas_frames, axis=axis).raw
    polars_result = DF.concat(polars_frames, axis=axis).raw

    if axis == 0:
        assert pandas_result["a"].tolist() == [1, 2]
        assert polars_result["a"].to_list() == [1, 2]
    else:
        assert pandas_result.iloc[0].tolist() == [1, 2]
        assert polars_result.row(0) == (1, 2)


def test_merge_asof_pandas_and_polars():
    pl = pytest.importorskip("polars")
    left = pd.DataFrame({"time": [1, 5, 10], "left": ["a", "b", "c"]})
    right = pd.DataFrame({"time": [2, 6], "right": ["x", "y"]})
    pandas_result = DF(left).merge_asof(right, on="time").raw
    polars_result = DF(pl.DataFrame({"time": [1, 5, 10], "left": ["a", "b", "c"]}))
    polars_result = polars_result.merge_asof(
        pl.DataFrame({"time": [2, 6], "right": ["x", "y"]}), on="time"
    ).raw

    assert pd.isna(pandas_result["right"].iloc[0])
    assert pandas_result["right"].iloc[1:].tolist() == ["x", "y"]
    assert polars_result["right"].to_list() == [None, "x", "y"]


def test_cross_join_pandas_and_polars():
    pl = pytest.importorskip("polars")
    pandas_result = (
        DF(pd.DataFrame({"a": [1, 2]})).cross_join(pd.DataFrame({"b": ["x", "y"]})).raw
    )
    polars_result = (
        DF(pl.DataFrame({"a": [1, 2]})).cross_join(pl.DataFrame({"b": ["x", "y"]})).raw
    )

    assert len(pandas_result) == len(polars_result) == 4
    assert pandas_result.iloc[0].tolist() == [1, "x"]
    assert polars_result.row(0) == (1, "x")


@pytest.mark.parametrize(
    "method, expected", [("semi_join", [2]), ("anti_join", [1, 3])]
)
def test_semi_and_anti_join_pandas_and_polars(method, expected):
    pl = pytest.importorskip("polars")
    pandas_left = pd.DataFrame({"key": [1, 2, 3], "value": ["a", "b", "c"]})
    pandas_right = pd.DataFrame({"key": [2, 2]})
    polars_left = pl.DataFrame({"key": [1, 2, 3], "value": ["a", "b", "c"]})
    polars_right = pl.DataFrame({"key": [2, 2]})

    pandas_result = getattr(DF(pandas_left), method)(pandas_right, on="key").raw
    polars_result = getattr(DF(polars_left), method)(polars_right, on="key").raw

    assert pandas_result["key"].tolist() == expected
    assert polars_result["key"].to_list() == expected


def test_apply_pandas_only():
    source = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
    result = DF(source).apply(lambda column: column.sum())

    assert result.to_dict() == {"x": 3, "y": 30}

    pl = pytest.importorskip("polars")
    with pytest.raises(NotImplementedError, match="only supported for pandas"):
        DF(pl.DataFrame({"x": [1, 2]})).apply(lambda column: column.sum())


def test_pipe_pandas_and_polars():
    pl = pytest.importorskip("polars")
    pandas_result = DF(pd.DataFrame({"value": [1, 2]})).pipe(
        lambda frame: frame.assign(double=frame["value"] * 2)
    )
    polars_result = DF(pl.DataFrame({"value": [1, 2]})).pipe(
        lambda frame: frame.with_columns((pl.col("value") * 2).alias("double"))
    )

    assert pandas_result.raw["double"].tolist() == [2, 4]
    assert polars_result.raw["double"].to_list() == [2, 4]


@pytest.mark.parametrize(
    ("method", "kwargs", "expected"),
    [
        ("clip", {"lower": 2, "upper": 4}, [2, 2.35, 4]),
        ("abs", {}, [1.0, 2.35, 9.0]),
        ("round", {"decimals": 1}, [-1.0, 2.4, 9.0]),
        ("log", {"base": 10}, [0.0, math.log10(2.35), math.log10(9.0)]),
        ("sqrt", {}, [1.0, 2.35**0.5, 3.0]),
        ("pow", {"exponent": 2}, [1.0, 2.35**2, 81.0]),
    ],
)
def test_math_operations_pandas_and_polars(method, kwargs, expected):
    pl = pytest.importorskip("polars")
    values = [-1.0, 2.35, 9.0] if method in ("clip", "round") else [1.0, 2.35, 9.0]
    pandas_source = pd.DataFrame({"value": values, "label": ["a", "b", "c"]})
    polars_source = pl.DataFrame({"value": values, "label": ["a", "b", "c"]})

    pandas_result = getattr(DF(pandas_source), method)(**kwargs).raw
    polars_result = getattr(DF(polars_source), method)(**kwargs).raw

    assert pandas_result["value"].tolist() == pytest.approx(expected)
    assert polars_result["value"].to_list() == pytest.approx(expected)
    assert pandas_result["label"].tolist() == ["a", "b", "c"]
    assert polars_result["label"].to_list() == ["a", "b", "c"]


def test_log_rejects_invalid_base():
    with pytest.raises(ValueError, match="base must be"):
        DF(pd.DataFrame({"value": [1]})).log(base=1)


def test_isin_and_between_pandas_polars():
    pl = pytest.importorskip("polars")
    pandas_source = pd.DataFrame({"value": [1, 2, 3]})
    polars_source = pl.DataFrame({"value": [1, 2, 3]})

    assert DF(pandas_source).isin([1, 3]).raw["value"].tolist() == [
        True,
        False,
        True,
    ]
    assert DF(polars_source).isin([1, 3]).raw["value"].to_list() == [
        True,
        False,
        True,
    ]
    assert DF(pandas_source).between("value", 2, 3).raw["value"].tolist() == [2, 3]
    assert DF(polars_source).between("value", 2, 3).raw["value"].to_list() == [2, 3]


def test_statistics_and_windows_accept_column_expressions(numeric_df):
    @enable(numeric_df)
    def result(df):
        return df.shift(price)  # noqa: F821

    assert result(numeric_df).raw["price"].iloc[1:].tolist() == [50, 150, 200]


def test_getattr_on_uninitialized_df_raises_cleanly():
    obj = DF.__new__(DF)
    with pytest.raises(AttributeError, match="uninitialized"):
        _ = obj.some_attr


def test_getattr_missing_attribute_raised_df_error():
    df = DF(pd.DataFrame({"a": [1]}))
    with pytest.raises(AttributeError, match="'DF' object has no attribute 'nope'"):
        _ = df.nope


def test_setattr_on_slots_does_not_recurse():
    df = DF(pd.DataFrame({"a": [1]}))
    df._cache = {"k": "v"}
    assert df._cache == {"k": "v"}
