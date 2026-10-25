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
from .enable import enable


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

        # Wrap the df and compile the cell with our enable tracer
        wrapped = DF(df)

        # We use the same closure trick: create a function and inject the wrapped df.
        # But since this is a cell, we need to execute it with the tracer active.
        import sys

        old_trace = sys.gettrace()

        def tracer(frame, event, arg):
            if event == "call":
                # Only inject if it's the exec frame for our cell
                if frame.f_code.co_filename == "<dfmacro-magic>":
                    from .expr import ColumnExpr
                    from .injection import get_strategy, inject_or_raise

                    strategy = get_strategy()
                    local_map = {}
                    # Inject all df columns
                    for col in df.columns:
                        local_map[col] = ColumnExpr(col)

                    local_map[df_name] = wrapped

                    inject_or_raise(strategy, frame, local_map)
                    return _make_local_tracer(strategy, frame, list(local_map))
            return tracer

        def _make_local_tracer(strategy, target_frame, injected_names):
            def local_tracer(frame, event, arg):
                if frame is target_frame and event in ("return", "exception"):
                    strategy.cleanup(frame, injected_names)
                return local_tracer

            return local_tracer

        sys.settrace(tracer)
        try:
            # Execute the cell code in the user's namespace
            # We wrap it in a code object with the proper filename for tracing
            code = compile(cell, "<dfmacro-magic>", "exec")
            exec(code, user_ns)
        finally:
            sys.settrace(old_trace)


def load_ipython_extension(ipython: Any) -> None:
    """Register the magic with IPython."""
    ipython.register_magics(DfmacroMagics)
