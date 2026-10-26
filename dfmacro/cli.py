"""Command Line Interface (CLI) for dfmacro."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dfmacro",
        description="Native Python syntax for DataFrame expressions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: dfmacro run <script.py> <data.csv>
    run_parser = subparsers.add_parser(
        "run", help="Run a Python script on a CSV/Parquet file"
    )
    run_parser.add_argument("script", type=str, help="Path to the Python script")
    run_parser.add_argument("data", type=str, help="Path to the CSV or Parquet file")

    # Command: dfmacro stub <data.csv> <output.pyi>
    stub_parser = subparsers.add_parser(
        "stub", help="Generate .pyi type hints for a data file"
    )
    stub_parser.add_argument("data", type=str, help="Path to the CSV or Parquet file")
    stub_parser.add_argument(
        "--out", type=str, default="dfmacro_stubs.pyi", help="Output .pyi filename"
    )

    args = parser.parse_args()

    if args.command == "run":
        _run_script(args.script, args.data)
    elif args.command == "stub":
        _generate_stub(args.data, args.out)


def _load_data(path: str) -> pd.DataFrame:
    """Load data based on file extension."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .csv or .parquet")


def _run_script(script_path: str, data_path: str) -> None:
    """Execute a Python script, injecting the loaded DataFrame as 'df'."""
    df = _load_data(data_path)

    script = Path(script_path)
    if not script.exists():
        print(f"Error: Script {script_path} not found.")
        sys.exit(1)

    print(f"🚀 Running {script_path} with data from {data_path}...")

    # Prepare the namespace
    namespace = {
        "df": df,
        "__name__": "__main__",
        "__file__": str(script),
    }

    # Execute the script
    try:
        code = compile(script.read_text(), str(script), "exec")
        exec(code, namespace)
    except Exception as e:
        print(f"❌ Error executing script: {e}")
        sys.exit(1)


def _generate_stub(data_path: str, output_path: str) -> None:
    """Load data and generate the .pyi stub."""
    from .ide import write_stub_to_file

    df = _load_data(data_path)
    write_stub_to_file(df, output_path)


if __name__ == "__main__":
    main()
