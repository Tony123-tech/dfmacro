"""Tests for the CLI."""

from __future__ import annotations

import pandas as pd
import pytest

from dfmacro.cli import _generate_stub, _load_data


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"price": [50, 150], "category": ["a", "b"]}).to_csv(path, index=False)
    return str(path)


def test_cli_load_data(csv_file):
    df = _load_data(csv_file)
    assert "price" in df.columns
    assert len(df) == 2


def test_cli_generate_stub(csv_file, tmp_path):
    out_file = tmp_path / "test_stubs.pyi"
    _generate_stub(csv_file, str(out_file))

    assert out_file.exists()
    content = out_file.read_text()
    assert "price: Series" in content
