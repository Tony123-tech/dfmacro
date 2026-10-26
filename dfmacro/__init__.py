"""dfmacro: native Python syntax for DataFrame expressions."""

from .compiler import compile_to_sql
from .df_wrapper import DF
from .enable import enable
from .expr import ColumnExpr, Literal, when
from .ide import write_stub_to_file

__version__ = "0.0.1"
__all__ = ["DF", "ColumnExpr", "Literal", "compile_to_sql", "enable", "when", "write_stub_to_file"]


def load_ipython_extension(ipython):
    from .magic import load_ipython_extension as _load

    _load(ipython)
