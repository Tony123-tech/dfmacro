"""
Tests for the data profiling and test generation module.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from dfmacro import (
    DF,
    generate_pytest_code,
    profile,
    save_profile_report,
    validate_dataframe,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, None, 28],
            "salary": [50000.0, 60000.0, 75000.0, 55000.0, 65000.0],
            "department": ["HR", "Engineering", "Engineering", "HR", "Marketing"],
            "active": [True, True, False, True, True],
        }
    )


@pytest.fixture
def numeric_df():
    """Create a numeric-only DataFrame for testing."""
    return pd.DataFrame(
        {
            "price": [10.5, 20.3, 15.7, 25.1, 18.9],
            "quantity": [5, 3, 7, 2, 4],
            "discount": [0.1, 0.0, 0.2, 0.15, 0.05],
        }
    )


def test_profile_basic(sample_df):
    """Test basic profiling functionality."""
    result = profile(sample_df)

    # Check basic structure
    assert "columns" in result
    assert "dtypes" in result
    assert "null_counts" in result
    assert "unique_counts" in result
    assert "numeric_stats" in result
    assert "categorical_stats" in result
    assert "shape" in result

    # Check values
    assert result["shape"] == (5, 6)
    assert set(result["columns"]) == {
        "id",
        "name",
        "age",
        "salary",
        "department",
        "active",
    }
    assert result["dtypes"]["id"] == "int64"
    # String dtype can be 'object' or 'str' depending on pandas version
    assert result["dtypes"]["name"] in ["object", "str"]
    assert result["dtypes"]["age"] == "float64"
    assert result["null_counts"]["age"] == 1
    assert result["null_counts"]["id"] == 0
    assert result["unique_counts"]["department"] == 3
    assert result["unique_counts"]["active"] == 2


def test_profile_numeric_stats(sample_df):
    """Test numeric statistics in profiling."""
    result = profile(sample_df)

    # Check numeric columns have stats
    assert "salary" in result["numeric_stats"]
    assert "age" in result["numeric_stats"]
    assert "id" in result["numeric_stats"]

    salary_stats = result["numeric_stats"]["salary"]
    assert "min" in salary_stats
    assert "max" in salary_stats
    assert "mean" in salary_stats
    assert "std" in salary_stats

    # Check actual values
    assert salary_stats["min"] == 50000.0
    assert salary_stats["max"] == 75000.0
    assert 55000.0 < salary_stats["mean"] < 65000.0


def test_profile_categorical_stats(sample_df):
    """Test categorical statistics in profiling."""
    result = profile(sample_df)

    # Check categorical columns have stats
    assert "department" in result["categorical_stats"]
    assert "name" in result["categorical_stats"]

    dept_stats = result["categorical_stats"]["department"]
    assert "top_values" in dept_stats
    assert "value_counts_total" in dept_stats

    # Engineering should appear twice
    assert dept_stats["top_values"].get("Engineering", 0) == 2
    assert dept_stats["value_counts_total"] == 5


def test_df_wrapper_profile(sample_df):
    """Test profile method on DF wrapper."""
    df = DF(sample_df)
    result = df.profile()

    assert result["shape"] == (5, 6)
    assert "salary" in result["numeric_stats"]
    assert result["null_counts"]["age"] == 1


def test_generate_pytest_code(sample_df):
    """Test pytest code generation."""
    code = generate_pytest_code(sample_df, "test_sample.py", "test_my_dataframe")

    # Check basic structure
    assert "import pytest" in code
    assert "import pandas as pd" in code
    assert "def test_my_dataframe():" in code
    assert "assert 'id' in df.columns" in code
    assert "assert 'name' in df.columns" in code
    assert "assert df.shape == (5, 6)" in code

    # Check column assertions
    assert "assert str(df['id'].dtype) == 'int64'" in code
    assert "assert df['age'].isna().sum() == 1" in code
    assert "assert df['department'].nunique() == 3" in code

    # Check numeric constraints
    assert "assert df['salary'].min() >= 50000.0" in code
    assert "assert df['salary'].max() <= 75000.0" in code


def test_generate_pytest_code_with_numeric_df(numeric_df):
    """Test pytest code generation with numeric DataFrame."""
    code = generate_pytest_code(numeric_df, "test_numeric.py", "test_numeric_data")

    assert "assert 'price' in df.columns" in code
    assert "assert 'quantity' in df.columns" in code
    assert "assert 'discount' in df.columns" in code

    # Check numeric constraints
    assert "assert df['price'].min() >= 10.5" in code
    assert "assert df['price'].max() <= 25.1" in code
    assert "(df['price'] >= 0).all()" in code  # Should add non-negative check


def test_df_wrapper_generate_pytest_code(sample_df):
    """Test generate_pytest_code method on DF wrapper."""
    df = DF(sample_df)
    code = df.generate_pytest_code("test_wrapper.py", "test_wrapper_df")

    assert "def test_wrapper_df():" in code
    assert "assert 'id' in df.columns" in code
    assert "assert df.shape == (5, 6)" in code


def test_save_profile_report_json(sample_df):
    """Test saving profile report as JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        save_profile_report(sample_df, temp_path, format="json")

        # Check file exists and contains valid JSON
        assert os.path.exists(temp_path)
        with open(temp_path, "r") as f:
            data = json.load(f)

        assert "columns" in data
        assert "shape" in data
        assert data["shape"] == [5, 6]
    finally:
        os.unlink(temp_path)


def test_save_profile_report_txt(sample_df):
    """Test saving profile report as text."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        temp_path = f.name

    try:
        save_profile_report(sample_df, temp_path, format="txt")

        # Check file exists and contains expected content
        assert os.path.exists(temp_path)
        with open(temp_path, "r") as f:
            content = f.read()

        assert "DataFrame Profile Report" in content
        assert "Shape: (5, 6)" in content
        assert "id:" in content
        assert "name:" in content
        assert "salary:" in content
    finally:
        os.unlink(temp_path)


def test_save_profile_report_invalid_format(sample_df):
    """Test saving profile report with invalid format."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported format"):
            save_profile_report(sample_df, temp_path, format="xml")
    finally:
        os.unlink(temp_path)


def test_validate_dataframe_matching(sample_df):
    """Test validation when DataFrames match."""
    profile_data = profile(sample_df)
    results = validate_dataframe(sample_df, profile_data)

    assert len(results["errors"]) == 0
    # There might be warnings for dtype string differences


def test_validate_dataframe_different_columns():
    """Test validation when DataFrames have different columns."""
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [1, 2], "c": [5, 6]})  # Different column

    profile_data = profile(df1)
    results = validate_dataframe(df2, profile_data)

    # Check for missing columns (format can vary)
    error_msg = results["errors"][0]
    assert "Missing columns:" in error_msg
    assert "b" in error_msg

    # Check for extra columns (format can vary)
    warning_msg = results["warnings"][0]
    assert "Extra columns:" in warning_msg
    assert "c" in warning_msg


def test_validate_dataframe_different_shape():
    """Test validation when DataFrames have different shapes."""
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [1, 2], "b": [4, 5]})  # Fewer rows

    profile_data = profile(df1)
    results = validate_dataframe(df2, profile_data)

    assert "Shape mismatch" in results["warnings"][0]


def test_validate_dataframe_more_nulls():
    """Test validation when DataFrame has more nulls."""
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]})  # 1 null
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, 6]})  # 2 nulls

    profile_data = profile(df1)
    results = validate_dataframe(df2, profile_data)

    assert "has more nulls than expected" in results["warnings"][0]


def test_profile_with_polars():
    """Test profiling with Polars DataFrame if available."""
    try:
        import polars as pl

        pl_df = pl.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.3, 15.7]})

        result = profile(pl_df)
        assert result["shape"] == (3, 2)
        assert set(result["columns"]) == {"id", "value"}
        assert result["null_counts"]["id"] == 0

    except ImportError:
        pytest.skip("polars not installed")


def test_profile_with_dask():
    """Test profiling with Dask DataFrame if available."""
    try:
        import dask.dataframe as dd

        pd_df = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.3, 15.7]})
        dask_df = dd.from_pandas(pd_df, npartitions=1)

        result = profile(dask_df)
        assert result["shape"] == (3, 2)
        assert set(result["columns"]) == {"id", "value"}

    except ImportError:
        pytest.skip("dask not installed")


def test_generated_pytest_code_executable(sample_df):
    """Test that generated pytest code is syntactically valid and executable."""
    code = generate_pytest_code(sample_df, "test_executable.py", "test_executable_df")

    # Check imports are present
    assert "import pytest" in code
    assert "import pandas as pd" in code
    assert "import numpy as np" in code

    # Check function definition
    assert "def test_executable_df():" in code

    # Try to compile the code to check syntax
    try:
        compile(code, "test_executable.py", "exec")
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax error: {e}")

    # Create a temporary test file and try to run it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Replace the placeholder with actual DataFrame
        with open(temp_path, "r") as f:
            content = f.read()

        # Insert actual DataFrame creation
        content = content.replace(
            "    # df = ... # Your DataFrame loading code here",
            f"    import pandas as pd\n    df = pd.DataFrame({sample_df.to_dict()})",
        )

        with open(temp_path, "w") as f:
            f.write(content)

        # Try to import and run the test
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("test_module", temp_path)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
            # If we get here, the module loaded successfully
            assert hasattr(module, "test_executable_df")
        except Exception as e:
            pytest.fail(f"Generated test module failed to load: {e}")

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
