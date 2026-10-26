"""Cross-backend coverage for string and datetime accessors."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from dfmacro.compiler import compile_expr
from dfmacro import enable
from dfmacro.expr import (
    ColumnExpr,
    DatePartExpr,
    StringMethodExpr,
)


def _evaluate(frame, expression):
    result = compile_expr(expression, frame)
    if isinstance(frame, pd.DataFrame):
        return result.tolist()
    return frame.select(result.alias("result"))["result"].to_list()


@pytest.fixture
def string_frames():
    values = ["Hello World", "  abc  ", "123", "a1", "", "ABC"]
    return (
        pd.DataFrame({"text": values}),
        pl.DataFrame({"text": values}),
    )


@pytest.fixture
def datetime_frames():
    values = [
        "2024-02-29 12:34:56.123456789",
        "2023-12-31 23:30:00",
        "2024-01-01 00:00:00",
    ]
    pandas = pd.DataFrame({"date": pd.to_datetime(values, format="mixed")})
    polars = pl.DataFrame({"date": values}).with_columns(
        pl.col("date").str.to_datetime(time_unit="ns")
    )
    return pandas, polars


@pytest.mark.parametrize(
    ("method", "args", "kwargs", "expected"),
    [
        ("contains", ("o",), {}, [True, False, False, False, False, False]),
        ("startswith", ("  ",), {}, [False, True, False, False, False, False]),
        ("endswith", ("1",), {}, [False, False, False, True, False, False]),
        ("lower", (), {}, ["hello world", "  abc  ", "123", "a1", "", "abc"]),
        ("upper", (), {}, ["HELLO WORLD", "  ABC  ", "123", "A1", "", "ABC"]),
        ("strip", (), {}, ["Hello World", "abc", "123", "a1", "", "ABC"]),
        ("lstrip", (), {}, ["Hello World", "abc  ", "123", "a1", "", "ABC"]),
        ("rstrip", (), {}, ["Hello World", "  abc", "123", "a1", "", "ABC"]),
        ("len", (), {}, [11, 7, 3, 2, 0, 3]),
        (
            "replace",
            ("l", "X"),
            {},
            ["HeXXo WorXd", "  abc  ", "123", "a1", "", "ABC"],
        ),
        (
            "split",
            (" ",),
            {},
            [["Hello", "World"], ["", "", "abc", "", ""], ["123"], ["a1"], [""], ["ABC"]],
        ),
        ("slice", (1, 3), {}, ["ell", " ab", "23", "1", "", "BC"]),
        (
            "pad",
            (14,),
            {"side": "right", "fillchar": "_"},
            ["Hello World___", "  abc  _______", "123___________", "a1____________", "______________", "ABC___________"],
        ),
        ("zfill", (5,), {}, ["Hello World", "  abc  ", "00123", "000a1", "00000", "00ABC"]),
        ("match", (r"\d+",), {}, [False, False, True, False, False, False]),
        (
            "extract",
            (r"([A-Za-z]+)",),
            {},
            ["Hello", "abc", None, "a", None, "ABC"],
        ),
        (
            "title",
            (),
            {},
            ["Hello World", "  Abc  ", "123", "A1", "", "Abc"],
        ),
        ("count", ("l",), {}, [3, 0, 0, 0, 0, 0]),
        (
            "reverse",
            (),
            {},
            ["dlroW olleH", "  cba  ", "321", "1a", "", "CBA"],
        ),
        ("fullmatch", (r"\d+",), {}, [False, False, True, False, False, False]),
        ("isalnum", (), {}, [False, False, True, True, False, True]),
        ("isalpha", (), {}, [False, False, False, False, False, True]),
        ("isdigit", (), {}, [False, False, True, False, False, False]),
        ("islower", (), {}, [False, True, False, True, False, False]),
        ("isupper", (), {}, [False, False, False, False, False, True]),
    ],
)
def test_string_accessor_methods(
    string_frames, method, args, kwargs, expected
):
    pandas_frame, polars_frame = string_frames
    expression = getattr(ColumnExpr("text").str, method)(*args, **kwargs)

    assert isinstance(expression, StringMethodExpr)
    pandas_result = _evaluate(pandas_frame, expression)
    polars_result = _evaluate(polars_frame, expression)
    if method == "extract":
        pandas_result = [None if pd.isna(value) else value for value in pandas_result]
    assert pandas_result == expected
    assert polars_result == expected


def test_string_pad_center_and_validation():
    expression = ColumnExpr("text").str.pad(6, side="both", fillchar="-")
    pandas = pd.DataFrame({"text": ["a", "abcde"]})
    polars = pl.DataFrame({"text": ["a", "abcde"]})

    assert _evaluate(pandas, expression) == ["--a---", "abcde-"]
    assert _evaluate(polars, expression) == ["--a---", "abcde-"]
    with pytest.raises(ValueError, match="side must be"):
        ColumnExpr("text").str.pad(5, side="center")
    with pytest.raises(ValueError, match="fillchar must"):
        ColumnExpr("text").str.pad(5, fillchar="xx")
    with pytest.raises(ValueError, match="positive, 1-based"):
        ColumnExpr("text").str.extract(r"(.*)", group=0)


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        ("year", [2024, 2023, 2024]),
        ("month", [2, 12, 1]),
        ("day", [29, 31, 1]),
        ("hour", [12, 23, 0]),
        ("minute", [34, 30, 0]),
        ("second", [56, 0, 0]),
        ("microsecond", [123456, 0, 0]),
        ("nanosecond", [789, 0, 0]),
        ("dayofweek", [3, 6, 0]),
        ("quarter", [1, 4, 1]),
        ("dayofyear", [60, 365, 1]),
        ("week", [9, 52, 1]),
        ("days_in_month", [29, 31, 31]),
        ("is_month_start", [False, False, True]),
        ("is_month_end", [True, True, False]),
        ("is_year_start", [False, False, True]),
        ("is_year_end", [False, True, False]),
        ("is_leap_year", [True, False, True]),
        ("month_name", ["February", "December", "January"]),
        ("day_name", ["Thursday", "Sunday", "Monday"]),
    ],
)
def test_datetime_accessor_properties(datetime_frames, part, expected):
    pandas_frame, polars_frame = datetime_frames
    expression = getattr(ColumnExpr("date").dt, part)

    assert isinstance(expression, DatePartExpr)
    assert _evaluate(pandas_frame, expression) == expected
    assert _evaluate(polars_frame, expression) == expected


@pytest.mark.parametrize(
    ("method", "args", "expected"),
    [
        ("strftime", ("%Y-%m-%d",), ["2024-02-29", "2023-12-31", "2024-01-01"]),
        (
            "floor",
            ("1h",),
            [
                datetime(2024, 2, 29, 12),
                datetime(2023, 12, 31, 23),
                datetime(2024, 1, 1),
            ],
        ),
        (
            "ceil",
            ("1h",),
            [
                datetime(2024, 2, 29, 13),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
            ],
        ),
        (
            "round",
            ("1h",),
            [
                datetime(2024, 2, 29, 13),
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
            ],
        ),
        (
            "normalize",
            (),
            [
                datetime(2024, 2, 29),
                datetime(2023, 12, 31),
                datetime(2024, 1, 1),
            ],
        ),
    ],
)
def test_datetime_accessor_methods(datetime_frames, method, args, expected):
    pandas_frame, polars_frame = datetime_frames
    expression = getattr(ColumnExpr("date").dt, method)(*args)

    assert isinstance(expression, DatePartExpr)
    pandas_result = _evaluate(pandas_frame, expression)
    polars_result = _evaluate(polars_frame, expression)
    if method == "strftime":
        assert pandas_result == expected
        assert polars_result == expected
        return
    assert [pd.Timestamp(value).to_pydatetime() for value in pandas_result] == expected
    assert [pd.Timestamp(value).to_pydatetime() for value in polars_result] == expected


def test_accessor_proxies_and_nodes_use_slots():
    text = ColumnExpr("text").str
    date = ColumnExpr("date").dt

    assert not hasattr(text, "__dict__")
    assert not hasattr(date, "__dict__")
    assert not hasattr(text.contains("x"), "__dict__")
    assert not hasattr(date.year, "__dict__")


def test_expanded_accessors_work_with_enable():
    pandas_strings = pd.DataFrame({"text": ["alpha", "123", "a1"]})

    @enable(pandas_strings)
    def pandas_string_filter(df):
        return df[text.str.isalpha()]  # noqa: F821

    assert pandas_string_filter(pandas_strings).raw["text"].tolist() == ["alpha"]

    polars_dates = pl.DataFrame({"date": ["2024-02-29", "2024-02-28"]}).with_columns(
        pl.col("date").str.to_date()
    )

    @enable(polars_dates)
    def polars_date_filter(df):
        return df[date.dt.is_month_end]  # noqa: F821

    assert (
        polars_date_filter(polars_dates).raw["date"].to_list()
        == [datetime(2024, 2, 29).date()]
    )
