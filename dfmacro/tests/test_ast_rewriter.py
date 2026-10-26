"""Tests for the AST rewriter and local-name collection."""

from __future__ import annotations

import pandas as pd
import pytest

from dfmacro import enable
from dfmacro.ast_rewriter import _collect_local_names, rewrite_function


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "price": [50, 150, 200, 80],
            "category": ["books", "electronics", "electronics", "books"],
        }
    )


# --- ColumnRewriter behaviour -------------------------------------------------


def test_simple_filter_still_works(df):
    @enable(df)
    def result(df):
        return df[price > 100]  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [150, 200]


def test_local_assignment_shadows_column(df):
    """A local assignment must shadow the column name, not be rewritten."""

    @enable(df)
    def result(df):
        price = 100  # local int shadows the column
        return df[price > 50]  # 100 > 50 -> True; df[True] raises KeyError

    with pytest.raises(KeyError):
        result(df)


def test_local_variable_used_in_expression(df):
    @enable(df)
    def result(df):
        threshold = 100
        return df[price > threshold]  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [150, 200]


def test_for_loop_target_not_rewritten(df):
    @enable(df)
    def result(df):
        # `category` is a loop variable (str), not the column.
        # If it were rewritten to __dfmacro_df__["category"], then
        # `category == "books"` would be a filter expression. Since it
        # is a local str, `df[category]` looks up column "books" -> KeyError.
        for category in ["books"]:
            return df[category]

    with pytest.raises(KeyError):
        result(df)


def test_comprehension_target_not_rewritten(df):
    @enable(df)
    def result(df):
        cats = [c for c in ["books"]]
        # `category` here is the *column* (not in comprehension scope),
        # so this should filter correctly.
        return df[df.category == cats[0]]

    out = result(df)
    assert len(out) == 2
    assert list(out.raw["category"]) == ["books", "books"]


def test_walrus_target_not_rewritten(df):
    @enable(df)
    def result(df):
        return df[(threshold := 100) < price]  # noqa: F821, F841

    out = result(df)
    assert list(out.raw["price"]) == [150, 200]


def test_with_as_target_not_rewritten(df, tmp_path):
    @enable(df)
    def result(df):
        with open(tmp_path / "x.txt", "w") as fh:
            fh.write("hi")
        return df[price > 100]  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [150, 200]


def test_except_as_target_not_rewritten(df):
    @enable(df)
    def result(df):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            msg = str(exc)
        # `price` is a column; `len(msg)` is an int.
        return df.mutate(note=price * 0 + len(msg))  # noqa: F821

    out = result(df)
    assert "note" in out.columns


def test_import_name_not_rewritten(df):
    @enable(df)
    def result(df):
        import math

        return df.mutate(sqrt_price=price.pow(0.5) * math.sqrt(4))  # noqa: F821

    out = result(df)
    assert "sqrt_price" in out.columns


def test_nested_def_name_not_rewritten(df):
    @enable(df)
    def result(df):
        def price():
            return 42

        # `price` resolves to the nested function, not the column.
        return df.mutate(x=price() + price())

    out = result(df)
    assert "x" in out.columns
    assert list(out.raw["x"]) == [84, 84, 84, 84]


def test_global_declaration_allows_column_rewrite(df):
    @enable(df)
    def result(df):
        global price  # noqa: F824, PLW0602

        return df[price > 100]  # noqa: F821

    out = result(df)
    assert list(out.raw["price"]) == [150, 200]


# --- _collect_local_names unit tests -----------------------------------------


def test_collect_local_names_includes_signature_and_body():
    import ast
    import textwrap

    src = textwrap.dedent("""
        def f(df, threshold=100, *args, **kwargs):
            x = 1
            for i in range(3):
                pass
            with open("f") as fh:
                pass
            try:
                pass
            except ValueError as exc:
                pass
            return [c for c in range(3)]
        """)
    func_def = ast.parse(src).body[0]
    names = _collect_local_names(func_def)

    for expected in {
        "df",
        "threshold",
        "args",
        "kwargs",
        "x",
        "i",
        "fh",
        "exc",
        "c",
    }:
        assert expected in names


def test_collect_local_names_excludes_global_and_nonlocal():
    import ast
    import textwrap

    src = textwrap.dedent("""
        def f(df):
            global g
            g = 1
        """)
    func_def = ast.parse(src).body[0]
    names = _collect_local_names(func_def)

    assert "g" not in names
    assert "df" in names


# --- rewrite_function error paths --------------------------------------------


def test_rewrite_function_rejects_lambda():
    # A lambda has retrievable source but is not a FunctionDef, so the
    # rewriter raises TypeError.
    with pytest.raises(TypeError, match="Could not find function definition"):
        rewrite_function(lambda df: df)


def test_rewrite_function_requires_df_param():
    def no_df_param(x):
        return x

    with pytest.raises(TypeError, match="named 'df' or 'data'"):
        rewrite_function(no_df_param)
