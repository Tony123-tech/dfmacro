"""IPython magic for dfmacro.

Usage in Jupyter:
    %load_ext dfmacro
    %dfmacro df

    # From this point on, in this cell, you can write:
    df[price > 100]
"""

from __future__ import annotations

from typing import Any

try:
    from IPython.core.magic import Magics, cell_magic, magics_class
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
except ImportError:
    # Allow importing dfmacro without IPython installed
    Magics = object  # type: ignore

    def magics_class(cls):
        return cls  # type: ignore

    def cell_magic(func):
        return func  # type: ignore

    def magic_arguments():
        return lambda f: f  # type: ignore

    def argument(*args, **kwargs):
        return lambda f: f  # type: ignore

    def parse_argstring(*args, **kwargs):
        pass  # type: ignore


from .df_wrapper import DF
from .dynamic import compile_dynamic_cell


@magics_class
class DfmacroMagics(Magics):
    """dfmacro IPython magics."""

    @magic_arguments()
    @argument("df_name", type=str, help="The name of the DataFrame variable to wrap.")
    @cell_magic
    def dfmacro(self, line: str, cell: str) -> Any:
        """Run a cell with bare DataFrame column names."""
        args = parse_argstring(self.dfmacro, line)
        df_name = args.df_name

        # Fetch the df from the user's namespace
        user_ns = self.shell.user_ns
        if df_name not in user_ns:
            raise NameError(f"Variable '{df_name}' not found in the namespace.")

        df = user_ns[df_name]

        exec_namespace = dict(user_ns)
        exec_namespace[df_name] = DF(df)
        code, exec_namespace, columns_name = compile_dynamic_cell(
            cell,
            df_name,
            list(df.columns),
            exec_namespace,
        )
        exec(code, exec_namespace, exec_namespace)

        for name, value in exec_namespace.items():
            if name not in {"__builtins__", columns_name, df_name}:
                user_ns[name] = value


def load_ipython_extension(ipython: Any) -> None:
    """Register the magic with IPython."""
    ipython.register_magics(DfmacroMagics)
