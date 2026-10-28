"""dfmacro: native Python syntax for DataFrame expressions."""

from .compiler import compile_to_sql
from .df_wrapper import DF
from .enable import enable
from .expr import ColumnExpr, Literal, when
from .ide import write_stub_to_file
from .plugins import register_accessor, register_backend
from .profiling import (
    generate_pytest_code,
    profile,
    save_profile_report,
    validate_dataframe,
)

__version__ = "0.0.1"
__all__ = [
    "DF",
    "ColumnExpr",
    "Literal",
    "compile_to_sql",
    "enable",
    "generate_pytest_code",
    "profile",
    "register_accessor",
    "register_backend",
    "save_profile_report",
    "validate_dataframe",
    "when",
    "write_stub_to_file",
]


def load_ipython_extension(ipython):
    from .magic import load_ipython_extension as _load

    _load(ipython)
