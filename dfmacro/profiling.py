"""
Data profiling and test generation module for dfmacro.

This module provides functions to profile DataFrames and generate pytest code
to validate schema and constraints.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def profile(df: Any) -> Dict[str, Any]:
    """
    Profile a DataFrame to extract schema and statistical information.

    Args:
        df: DataFrame (pandas, polars, or dask)

    Returns:
        Dictionary containing profiling information:
        - columns: list of column names
        - dtypes: dictionary mapping column names to data types
        - null_counts: dictionary mapping column names to null counts
        - unique_counts: dictionary mapping column names to unique value counts
        - numeric_stats: dictionary with min, max, mean, std for numeric columns
        - categorical_stats: dictionary with value counts for categorical columns

    Raises:
        ValueError: If df is not a recognized DataFrame type
    """
    # Convert to pandas for profiling if needed
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    elif hasattr(df, "compute"):
        df = df.compute()
    elif hasattr(df, "collect"):
        df = df.collect()

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Unsupported DataFrame type: {type(df)}")

    result = {
        "columns": list(df.columns),
        "dtypes": {},
        "null_counts": {},
        "unique_counts": {},
        "numeric_stats": {},
        "categorical_stats": {},
        "shape": df.shape,
        "memory_usage": (
            df.memory_usage(deep=True).sum() if hasattr(df, "memory_usage") else None
        ),
    }

    for col in df.columns:
        # Data type
        dtype = str(df[col].dtype)
        result["dtypes"][col] = dtype

        # Null count
        null_count = int(df[col].isna().sum())
        result["null_counts"][col] = null_count

        # Unique count
        unique_count = int(df[col].nunique())
        result["unique_counts"][col] = unique_count

        # Numeric statistics (excluding boolean columns)
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(
            df[col]
        ):
            try:
                numeric_info = {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": (
                        float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                    ),
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    "median": (
                        float(df[col].median())
                        if not pd.isna(df[col].median())
                        else None
                    ),
                }
                # Try to calculate quantiles, but skip if they fail (e.g., for boolean columns)
                try:
                    q25 = df[col].quantile(0.25)
                    numeric_info["q25"] = float(q25) if not pd.isna(q25) else None
                except (TypeError, ValueError):
                    numeric_info["q25"] = None

                try:
                    q75 = df[col].quantile(0.75)
                    numeric_info["q75"] = float(q75) if not pd.isna(q75) else None
                except (TypeError, ValueError):
                    numeric_info["q75"] = None

                result["numeric_stats"][col] = numeric_info
            except Exception:
                # Skip numeric stats if any calculation fails
                pass

        # Categorical/string statistics
        elif pd.api.types.is_string_dtype(df[col]) or isinstance(
            df[col].dtype, pd.CategoricalDtype
        ):
            if unique_count <= 100:  # Limit to avoid huge outputs
                value_counts = df[col].value_counts().head(10).to_dict()
                result["categorical_stats"][col] = {
                    "top_values": value_counts,
                    "value_counts_total": int(df[col].value_counts().sum()),
                }

    return result


def generate_pytest_code(
    df: Any, filename: str, test_name: str = "test_dataframe_schema"
) -> str:
    """
    Generate pytest code to validate DataFrame schema and constraints.

    Args:
        df: DataFrame to profile
        filename: Name of the test file to generate
        test_name: Name of the test function

    Returns:
        String containing pytest code

    Raises:
        ValueError: If df is not a recognized DataFrame type
    """
    profile_data = profile(df)

    lines = [
        "import pytest",
        "import pandas as pd",
        "import numpy as np",
        "",
        f"def {test_name}():",
        "    # Load or create the DataFrame to test",
        "    # df = ... # Your DataFrame loading code here",
        "",
        "    # Schema validation",
    ]

    # Add column existence assertions
    for col in profile_data["columns"]:
        lines.append(f"    assert '{col}' in df.columns")

    lines.append("")
    lines.append("    # Data type validation")
    for col, dtype in profile_data["dtypes"].items():
        # Clean up dtype string for assertion
        dtype_clean = (
            dtype.replace("dtype('", "")
            .replace("')", "")
            .replace("<", "")
            .replace(">", "")
        )
        lines.append(
            f"    assert str(df['{col}'].dtype) == '{dtype_clean}'  # Original dtype: {dtype}"
        )

    lines.append("")
    lines.append("    # Null count validation")
    for col, null_count in profile_data["null_counts"].items():
        lines.append(f"    assert df['{col}'].isna().sum() == {null_count}")

    lines.append("")
    lines.append("    # Unique value count validation")
    for col, unique_count in profile_data["unique_counts"].items():
        lines.append(f"    assert df['{col}'].nunique() == {unique_count}")

    lines.append("")
    lines.append("    # Numeric column constraints")
    for col, stats in profile_data["numeric_stats"].items():
        if stats["min"] is not None:
            lines.append(
                f"    assert df['{col}'].min() >= {stats['min']}  # Minimum value check"
            )
        if stats["max"] is not None:
            lines.append(
                f"    assert df['{col}'].max() <= {stats['max']}  # Maximum value check"
            )

        # Add range-based constraints for numeric columns
        min_val = stats["min"]
        max_val = stats["max"]
        if min_val is not None and max_val is not None:
            if min_val >= 0:
                lines.append(
                    f"    assert (df['{col}'] >= 0).all()  # Non-negative values"
                )
            elif max_val <= 0:
                lines.append(
                    f"    assert (df['{col}'] <= 0).all()  # Non-positive values"
                )

    lines.append("")
    lines.append("    # Shape validation")
    lines.append(f"    assert df.shape == {profile_data['shape']}")

    # Add sample data validation if available
    lines.append("")
    lines.append("    # Sample data validation (optional)")
    lines.append("    # Add specific value checks based on known data patterns")

    lines.append("")
    lines.append("    # Return True if all tests pass")
    lines.append("    return True")

    # Add docstring
    docstring = f'"""Test generated from profiling DataFrame with shape {profile_data["shape"]}"""'
    lines.insert(3, docstring)

    # Add header comment
    header = [
        f"# Auto-generated pytest test for DataFrame validation",
        f"# Generated from profiling a DataFrame with {len(profile_data['columns'])} columns",
        f"# Run with: pytest {filename}",
        "",
    ]

    return "\n".join(header + lines)


def save_profile_report(df: Any, output_path: str, format: str = "json") -> None:
    """
    Save profiling report to a file.

    Args:
        df: DataFrame to profile
        output_path: Path to save the report
        format: Output format ('json' or 'txt')

    Raises:
        ValueError: If format is not supported
    """
    profile_data = profile(df)

    if format == "json":
        import json

        with open(output_path, "w") as f:
            json.dump(profile_data, f, indent=2, default=str)
    elif format == "txt":
        with open(output_path, "w") as f:
            f.write(f"DataFrame Profile Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Shape: {profile_data['shape']}\n")
            if profile_data.get("memory_usage"):
                f.write(f"Memory Usage: {profile_data['memory_usage']} bytes\n")
            f.write("\nColumns:\n")
            for col in profile_data["columns"]:
                f.write(f"  - {col}\n")

            f.write("\nColumn Details:\n")
            for col in profile_data["columns"]:
                f.write(f"\n  {col}:\n")
                f.write(f"    Type: {profile_data['dtypes'][col]}\n")
                f.write(f"    Null count: {profile_data['null_counts'][col]}\n")
                f.write(f"    Unique values: {profile_data['unique_counts'][col]}\n")

                if col in profile_data["numeric_stats"]:
                    stats = profile_data["numeric_stats"][col]
                    f.write("    Numeric statistics:\n")
                    for stat_name, stat_value in stats.items():
                        if stat_value is not None:
                            f.write(f"      {stat_name}: {stat_value}\n")

                if col in profile_data["categorical_stats"]:
                    stats = profile_data["categorical_stats"][col]
                    f.write("    Categorical statistics:\n")
                    f.write(
                        f"      Total value counts: {stats['value_counts_total']}\n"
                    )
                    if "top_values" in stats:
                        f.write("      Top values:\n")
                        for value, count in stats["top_values"].items():
                            f.write(f"        {value}: {count}\n")
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_dataframe(df: Any, profile_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate a DataFrame against a profile.

    Args:
        df: DataFrame to validate
        profile_data: Profile data from the profile() function

    Returns:
        Dictionary with validation results:
        - 'errors': List of error messages
        - 'warnings': List of warning messages
    """
    results = {"errors": [], "warnings": []}

    try:
        current_profile = profile(df)
    except Exception as e:
        results["errors"].append(f"Failed to profile DataFrame: {str(e)}")
        return results

    # Check columns
    expected_columns = set(profile_data["columns"])
    current_columns = set(current_profile["columns"])

    missing_columns = expected_columns - current_columns
    extra_columns = current_columns - expected_columns

    if missing_columns:
        results["errors"].append(f"Missing columns: {sorted(missing_columns)}")

    if extra_columns:
        results["warnings"].append(f"Extra columns: {sorted(extra_columns)}")

    # Check data types for common columns
    common_columns = expected_columns & current_columns
    for col in common_columns:
        expected_dtype = profile_data["dtypes"][col]
        current_dtype = current_profile["dtypes"][col]

        # Compare cleaned dtype strings
        expected_clean = (
            expected_dtype.replace("dtype('", "")
            .replace("')", "")
            .replace("<", "")
            .replace(">", "")
        )
        current_clean = (
            current_dtype.replace("dtype('", "")
            .replace("')", "")
            .replace("<", "")
            .replace(">", "")
        )

        if expected_clean != current_clean:
            results["warnings"].append(
                f"Column '{col}' has different dtype: expected {expected_dtype}, got {current_dtype}"
            )

    # Check null counts for common columns
    for col in common_columns:
        expected_null = profile_data["null_counts"][col]
        current_null = current_profile["null_counts"][col]

        if current_null > expected_null:
            results["warnings"].append(
                f"Column '{col}' has more nulls than expected: expected {expected_null}, got {current_null}"
            )

    # Check shape
    if current_profile["shape"] != profile_data["shape"]:
        results["warnings"].append(
            f"Shape mismatch: expected {profile_data['shape']}, got {current_profile['shape']}"
        )

    return results
