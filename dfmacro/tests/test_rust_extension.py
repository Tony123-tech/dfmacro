# pyright: reportAttributeAccessIssue=false
"""Smoke tests for the dfmacro_rs Rust extension."""

import numpy as np
import pytest
from dfmacro_rs import (
    corr_parallel,
    cov_parallel,
    groupby_sum_parallel,
    math_scalar,
    mean_masked,
    nlargest_parallel,
    nsmallest_parallel,
    sort_parallel,
    std_parallel,
    sum_masked,
    text_contains_parallel,
    text_len_parallel,
    text_lower_parallel,
    text_replace_parallel,
    text_strip_parallel,
    text_upper_parallel,
    unique_parallel,
    value_counts_parallel,
    var_parallel,
)


def test_corr_cov_nan_handling():
    # all-NaN column should return NaN, not crash or inf
    a = np.array([np.nan, np.nan, np.nan])
    b = np.array([1.0, 2.0, 3.0])
    assert np.isnan(corr_parallel(a, b))
    assert np.isnan(cov_parallel(a, b))

    # only one valid pair should return NaN
    a = np.array([1.0, np.nan])
    b = np.array([2.0, np.nan])
    assert np.isnan(corr_parallel(a, b))
    assert np.isnan(cov_parallel(a, b))


def test_corr_cov_normal():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    assert abs(corr_parallel(a, b) - 1.0) < 1e-9
    assert abs(cov_parallel(a, b) - 2.0) < 1e-9


def test_unique_parallel():
    arr = np.array([1.0, 2.0, 2.0, 3.0, np.nan])
    out = unique_parallel(arr)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64

    finite = out[~np.isnan(out)]
    n_nan = int(np.isnan(out).sum())
    assert sorted(finite.tolist()) == [1.0, 2.0, 3.0]
    assert n_nan == 1
    assert len(out) == 4


def test_value_counts_parallel():
    vals, counts = value_counts_parallel(np.array([1.0, 2.0, 2.0, 3.0]))
    assert vals.dtype == np.float64
    assert counts.dtype == np.int64
    assert counts[0] == 2
    assert sorted(vals.tolist()) == [1.0, 2.0, 3.0]


def test_text_lower_upper():
    objs = np.array(["Hello", "WORLD", "  trim  "], dtype=object)
    assert list(text_lower_parallel(objs)) == ["hello", "world", "  trim  "]
    assert list(text_upper_parallel(objs)) == ["HELLO", "WORLD", "  TRIM  "]


def test_text_contains():
    objs = np.array(["Hello", "WORLD", "  trim  "], dtype=object)
    # Only "Hello" contains lowercase 'o'; "WORLD" has uppercase 'O'.
    assert list(text_contains_parallel(objs, "o")) == [True, False, False]


def test_text_len_replace_strip():
    objs = np.array(["Hello", "WORLD", "  trim  "], dtype=object)
    assert list(text_len_parallel(objs)) == [5, 5, 8]
    # "WORLD" has no lowercase 'o', so it is unchanged.
    assert list(text_replace_parallel(objs, "o", "0")) == [
        "Hell0",
        "WORLD",
        "  trim  ",
    ]
    assert list(text_strip_parallel(objs)) == ["Hello", "WORLD", "trim"]


def test_groupby_sum_parallel():
    out = groupby_sum_parallel(["a", "b", "a", "c"], np.array([1.0, 2.0, 3.0, 4.0]))
    # a=4.0, b=2.0, c=4.0 sorted ascending
    assert sorted(out.tolist()) == [2.0, 4.0, 4.0]


def test_math_scalar():
    arr = np.array([1.0, 2.0, 4.0])
    assert np.allclose(math_scalar(arr, "+", 2.0), [3.0, 4.0, 6.0])
    assert np.allclose(math_scalar(arr, "-", 1.0), [0.0, 1.0, 3.0])
    assert np.allclose(math_scalar(arr, "*", 2.0), [2.0, 4.0, 8.0])
    assert np.allclose(math_scalar(arr, "/", 2.0), [0.5, 1.0, 2.0])


def test_math_scalar_rejects_unknown_op():
    with pytest.raises(ValueError):
        math_scalar(np.array([1.0]), "???", 2.0)


def test_masked_reductions():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, False, True, False])

    assert sum_masked(arr, mask) == 4.0
    assert mean_masked(arr, mask) == 2.0
    assert abs(var_parallel(arr) - 5.0 / 3.0) < 1e-9
    assert abs(std_parallel(arr) - (5.0 / 3.0) ** 0.5) < 1e-9


def test_sorting():
    arr = np.array([3.0, 1.0, 4.0, 1.5, 5.0])

    assert list(sort_parallel(arr, False)) == [1.0, 1.5, 3.0, 4.0, 5.0]
    assert list(sort_parallel(arr, True)) == [5.0, 4.0, 3.0, 1.5, 1.0]
    assert list(nlargest_parallel(arr, 2)) == [5.0, 4.0]
    assert list(nsmallest_parallel(arr, 2)) == [1.0, 1.5]
