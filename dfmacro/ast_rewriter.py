"""AST rewriter for dfmacro.

This module replaces the sys.settrace approach by rewriting the user's
function source code at definition time. It transforms bare names like:

    df[price > 100]

into an expression tree:

    df[ColumnExpr("price") > 100]

This is much faster (zero runtime overhead) and works correctly in
multithreaded environments.
"""

from __future__ import annotations

import ast
import builtins
import functools
import linecache
import textwrap
from typing import Any, Callable, TypeVar, cast

from .expr import ColumnExpr

_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))
F = TypeVar("F", bound=Callable[..., Any])


class _StoreNameCollector(ast.NodeVisitor):
    """Collect every name that is *bound* inside the function body."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self._global_names: set[str] = set()
        self._nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_Global(self, node: ast.Global) -> None:
        self._global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._nonlocal_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self.names.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def finalize(self) -> set[str]:
        return self.names - self._global_names - self._nonlocal_names


class ColumnRewriter(ast.NodeTransformer):
    """Transform bare column names into expression-tree constructor calls."""

    df_var_name: str
    local_names: set[str]
    protected_names: frozenset[str] | set[str]

    def __init__(
        self,
        df_var_name: str,
        local_names: set[str],
        protected_names: frozenset[str] | set[str],
    ) -> None:
        self.df_var_name = df_var_name
        self.local_names = local_names
        self.protected_names = protected_names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load):
            if node.id == self.df_var_name:
                return node
            if node.id in self.local_names:
                return node
            if node.id in self.protected_names:
                return node
            if node.id in _BUILTIN_NAMES:
                return node
            return ast.Call(
                func=ast.Name(id="__dfmacro_column__", ctx=ast.Load()),
                args=[ast.Constant(value=node.id)],
                keywords=[],
            )
        return node


def _collect_local_names(func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect all names bound in the function signature and body."""
    local_names: set[str] = set()

    args = func_def.args
    for arg in args.posonlyargs:
        local_names.add(arg.arg)
    for arg in args.args:
        local_names.add(arg.arg)
    for arg in args.kwonlyargs:
        local_names.add(arg.arg)
    if args.vararg:
        local_names.add(args.vararg.arg)
    if args.kwarg:
        local_names.add(args.kwarg.arg)

    collector = _StoreNameCollector()
    for stmt in func_def.body:
        collector.visit(stmt)
    local_names |= collector.finalize()

    return local_names


def _read_function_source(func: Callable[..., Any]) -> str:
    """Read the source of a function starting at its ``def`` line."""
    code = func.__code__
    filename = code.co_filename
    def_lineno = code.co_firstlineno

    first_line = linecache.getline(filename, def_lineno)
    if not first_line.strip():
        raise RuntimeError(
            f"Cannot read source for {func.__name__} at " f"{filename}:{def_lineno}."
        )

    base_indent = len(first_line) - len(first_line.lstrip())

    lines: list[str] = []
    lineno = def_lineno
    while True:
        line = linecache.getline(filename, lineno)
        if not line:
            break
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            if indent < base_indent and lines:
                break
        lines.append(line)
        lineno += 1
        if lineno - def_lineno > 2000:
            break

    return textwrap.dedent("".join(lines))


def rewrite_function(func: F) -> F:
    """Rewrite a function's AST to inject column names."""
    try:
        source = _read_function_source(func)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot rewrite {func.__name__}: {exc} "
            "Interactive functions may not have retrievable source."
        ) from exc

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        raise SyntaxError(
            f"Cannot parse source for {func.__name__} at {location}: {exc.msg}"
        ) from exc

    func_def = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func.__name__:
                func_def = node
                break

    if func_def is None:
        raise TypeError(
            f"Could not find function definition for {func.__name__!r} in source."
        )

    func_def.decorator_list = []

    parameters = (
        func_def.args.posonlyargs
        + func_def.args.args
        + func_def.args.kwonlyargs
    )
    df_var_name = None
    for arg in parameters:
        if arg.arg in ("df", "data"):
            df_var_name = arg.arg
            break

    if not df_var_name:
        raise TypeError(
            f"Function {func.__name__} must accept a positional or keyword-only "
            "parameter named 'df' or 'data'."
        )

    local_names = _collect_local_names(func_def)

    protected: set[str] = set(func.__globals__)
    protected.update(func.__code__.co_freevars)

    rewriter = ColumnRewriter(df_var_name, local_names, protected)
    new_func_def = rewriter.visit(func_def)
    ast.fix_missing_locations(new_func_def)

    new_module = ast.Module(body=[new_func_def], type_ignores=[])
    ast.fix_missing_locations(new_module)
    compiled_code = compile(new_module, func.__code__.co_filename, "exec")

    # Build a namespace that resolves both module globals and closure
    # free variables. The rewritten function has no __closure__ (it was
    # compiled fresh), so any free variable in the original function must
    # be injected here as a global.
    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "__dfmacro_column__": ColumnExpr,
    }
    namespace.update(func.__globals__)
    if func.__closure__:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                namespace[name] = cell.cell_contents
            except ValueError:
                # Empty cell (e.g. `del`'d). Skip.
                pass

    exec(compiled_code, namespace)

    new_func = namespace[func.__name__]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return new_func(*args, **kwargs)

    return cast(F, wrapper)
