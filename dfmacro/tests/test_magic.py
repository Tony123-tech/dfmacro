"""Tests for the IPython magic."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("IPython")

from IPython.core.interactiveshell import InteractiveShell

from dfmacro.magic import DfmacroMagics


@pytest.fixture
def shell():
    """Create a mock IPython shell with a sample DataFrame."""
    shell = InteractiveShell.instance()
    df = pd.DataFrame(
        {
            "price": [50, 150, 200, 80],
            "category": ["books", "electronics", "electronics", "books"],
        }
    )
    shell.user_ns = {"df": df}
    return shell


def test_magic_basic_filter(shell):
    """Test that %dfmacro allows bare column names in a cell."""
    magic = DfmacroMagics(shell)
    cell = "result = df[price > 100]"

    magic.dfmacro("df", cell)

    assert "result" in shell.user_ns
    out = shell.user_ns["result"]
    assert len(out) == 2
    assert list(out.raw["price"]) == [150, 200]


def test_magic_cleanup(shell):
    """Test that the magic cleans up the injected column names after execution."""
    magic = DfmacroMagics(shell)
    cell = "result = df[price > 100]"

    magic.dfmacro("df", cell)

    # 'price' and 'category' should not be in the namespace anymore
    assert "price" not in shell.user_ns
    assert "category" not in shell.user_ns

    # 'df' should still be the original pandas DataFrame (not the wrapper)
    assert isinstance(shell.user_ns["df"], pd.DataFrame)


def test_magic_mutate(shell):
    """Test that %dfmacro supports mutate."""
    magic = DfmacroMagics(shell)
    cell = "result = df.mutate(double=price * 2)"

    magic.dfmacro("df", cell)

    out = shell.user_ns["result"]
    assert "double" in out.columns
    assert list(out.raw["double"]) == [100, 300, 400, 160]
