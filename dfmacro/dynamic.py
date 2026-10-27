"""Source-based function generation for dfmacro."""

from __future__ import annotations

import ast
import builtins
import functools
import inspect
import textwrap
import types
from collections.abc import Callable
from typing import Any

from .expr import ColumnExpr

_BUILTIN_NAMES = frozenset(dir(builtins))
_FRAME_ARGUMENT = "__dfmacro_frame__"


class _LocalNameCollector(ast.NodeVisitor):
    """Collect names local to one function scope."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.visit(node.iter)
        for expression in node.ifs:
            self.visit(expression)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.target)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.names.add(node.name)
        if node.pattern:
            self.visit(node.pattern)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.names.add(node.rest)
        self.generic_visit(node)


def _function_locals(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    arguments = function.args
    names = {
        arg.arg
        for arg in (
            arguments.posonlyargs
            + arguments.args
            + arguments.kwonlyargs
        )
    }
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)

    collector = _LocalNameCollector()
    for statement in function.body:
        collector.visit(statement)
    return names | (collector.names - collector.global_names - collector.nonlocal_names)


def _all_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _unique_name(base: str, names: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


class _ColumnSubscriptRewriter(ast.NodeTransformer):
    def __init__(
        self,
        columns_name: str,
        frame_name: str,
        frame_argument: str,
        columns: set[str],
        local_names: set[str],
        protected_names: set[str],
    ) -> None:
        self.columns_name = columns_name
        self.frame_name = frame_name
        self.frame_argument = frame_argument
        self.columns = columns
        self.scope_names = [local_names]
        self.protected_names = protected_names
        self._root_function: ast.AST | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if self._root_function is None:
            self._root_function = node
            return self.generic_visit(node)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self.scope_names.append(_function_locals(node))
        node.body = [self.visit(statement) for statement in node.body]
        self.scope_names.pop()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if self._root_function is None:
            self._root_function = node
            return self.generic_visit(node)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self.scope_names.append(_function_locals(node))
        node.body = [self.visit(statement) for statement in node.body]
        self.scope_names.pop()
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        names = {
            arg.arg
            for arg in (
                node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            )
        }
        if node.args.vararg:
            names.add(node.args.vararg.arg)
        if node.args.kwarg:
            names.add(node.args.kwarg.arg)
        self.scope_names.append(names)
        node.body = self.visit(node.body)
        self.scope_names.pop()
        return node

    def _visit_comprehension(self, node: ast.AST, value_fields: tuple[str, ...]) -> ast.AST:
        self.scope_names.append(set())
        for generator in node.generators:
            generator.iter = self.visit(generator.iter)
            target_names = {
                child.id
                for child in ast.walk(generator.target)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            }
            self.scope_names[-1].update(target_names)
            generator.ifs = [self.visit(expression) for expression in generator.ifs]
        for field in value_fields:
            setattr(node, field, self.visit(getattr(node, field)))
        self.scope_names.pop()
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._visit_comprehension(node, ("elt",))

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._visit_comprehension(node, ("elt",))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._visit_comprehension(node, ("elt",))

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self._visit_comprehension(node, ("key", "value"))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return node

        is_local = any(node.id in scope for scope in self.scope_names)
        if node.id == self.frame_name and not is_local:
            return ast.copy_location(
                ast.Name(id=self.frame_argument, ctx=ast.Load()), node
            )

        if node.id == self.columns_name or is_local:
            return node
        if node.id in self.protected_names or node.id in _BUILTIN_NAMES:
            return node
        if node.id not in self.columns and node.id.startswith("_"):
            return node

        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id=self.columns_name, ctx=ast.Load()),
                slice=ast.Constant(value=node.id),
                ctx=ast.Load(),
            ),
            node,
        )


class _CellNameRewriter(ast.NodeTransformer):
    def __init__(
        self,
        columns_name: str,
        frame_name: str,
        columns: set[str],
        local_names: set[str],
        protected_names: set[str],
    ) -> None:
        self.columns_name = columns_name
        self.frame_name = frame_name
        self.columns = columns
        self.scope_names = [local_names]
        self.protected_names = protected_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self.scope_names.append(_function_locals(node))
        node.body = [self.visit(statement) for statement in node.body]
        self.scope_names.pop()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self.scope_names.append(_function_locals(node))
        node.body = [self.visit(statement) for statement in node.body]
        self.scope_names.pop()
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        names = {
            arg.arg
            for arg in (
                node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            )
        }
        if node.args.vararg:
            names.add(node.args.vararg.arg)
        if node.args.kwarg:
            names.add(node.args.kwarg.arg)
        self.scope_names.append(names)
        node.body = self.visit(node.body)
        self.scope_names.pop()
        return node

    def _visit_comprehension(self, node: ast.AST, value_fields: tuple[str, ...]) -> ast.AST:
        self.scope_names.append(set())
        for generator in node.generators:
            generator.iter = self.visit(generator.iter)
            target_names = {
                child.id
                for child in ast.walk(generator.target)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            }
            self.scope_names[-1].update(target_names)
            generator.ifs = [self.visit(expression) for expression in generator.ifs]
        for field in value_fields:
            setattr(node, field, self.visit(getattr(node, field)))
        self.scope_names.pop()
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._visit_comprehension(node, ("elt",))

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._visit_comprehension(node, ("elt",))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._visit_comprehension(node, ("elt",))

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self._visit_comprehension(node, ("key", "value"))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return node
        if node.id == self.frame_name or any(
            node.id in scope for scope in self.scope_names
        ):
            return node
        if node.id in self.protected_names or node.id in _BUILTIN_NAMES:
            return node
        if node.id not in self.columns and node.id.startswith("_"):
            return node
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id=self.columns_name, ctx=ast.Load()),
                slice=ast.Constant(value=node.id),
                ctx=ast.Load(),
            ),
            node,
        )


class _ColumnNamespace:
    __slots__ = ()

    def __getitem__(self, name: str) -> ColumnExpr:
        return ColumnExpr(name)


def _parse_function(
    source: str, func: Callable[..., Any]
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(textwrap.dedent(source))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func.__name__
        ),
        None,
    )
    if function is None:
        raise RuntimeError(
            f"Could not locate function {func.__name__!r} in its source."
        )
    function.decorator_list = []
    return function


def compile_dynamic(
    func: Callable[..., Any],
    df_name: str,
    columns: list[str],
    *,
    frame_name: str | None = None,
) -> Callable[..., Any]:
    """Generate a function that resolves bare names through a lazy column namespace."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "dfmacro requires source code for AST rewriting. In Jupyter, please use "
            "the `%dfmacro` magic instead of the `@enable` decorator."
        ) from exc

    function = _parse_function(source, func)
    local_names = _function_locals(function)
    frame_name = frame_name or "df"
    all_names = _all_names(function) | local_names
    columns_name = _unique_name(df_name, all_names)
    frame_argument = _unique_name(_FRAME_ARGUMENT, all_names | {columns_name})

    protected_names = set(func.__globals__) | set(func.__code__.co_freevars)
    rewriter = _ColumnSubscriptRewriter(
        columns_name=columns_name,
        frame_name=frame_name,
        frame_argument=frame_argument,
        columns=set(columns),
        local_names=local_names,
        protected_names=protected_names,
    )
    function = rewriter.visit(function)
    function.decorator_list = []

    function.args.kwonlyargs.extend(
        [
            ast.arg(arg=columns_name),
            ast.arg(arg=frame_argument),
        ]
    )
    function.args.kw_defaults.extend([None, None])

    factory_name = _unique_name("__dfmacro_factory__", all_names)
    closure_names = list(func.__code__.co_freevars)
    factory_args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name) for name in closure_names],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    factory = ast.FunctionDef(
        name=factory_name,
        args=factory_args,
        body=[
            function,
            ast.Return(value=ast.Name(id=function.name, ctx=ast.Load())),
        ],
        decorator_list=[],
    )
    module = ast.Module(body=[factory], type_ignores=[])
    ast.fix_missing_locations(module)

    module_code = compile(module, func.__code__.co_filename, "exec")
    execution_globals = dict(func.__globals__)
    exec(module_code, execution_globals, execution_globals)
    factory_function = execution_globals[factory_name]
    closure_values = []
    for cell in func.__closure__ or ():
        try:
            closure_values.append(cell.cell_contents)
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot rewrite {func.__name__!r}: a closure variable is unbound."
            ) from exc
    provisional = factory_function(*closure_values)
    generated = types.FunctionType(
        provisional.__code__,
        func.__globals__,
        provisional.__name__,
        provisional.__defaults__,
        provisional.__closure__,
    )
    generated.__kwdefaults__ = provisional.__kwdefaults__
    generated.__name__ = func.__name__
    generated.__qualname__ = func.__qualname__
    generated.__doc__ = func.__doc__
    generated.__annotations__ = dict(getattr(func, "__annotations__", {}))
    generated.__dfmacro_columns_argument__ = columns_name
    generated.__dfmacro_frame_argument__ = frame_argument
    generated.__dfmacro_frame_name__ = frame_name
    generated.__dfmacro_columns_factory__ = _ColumnNamespace
    generated.__signature__ = inspect.signature(func)
    functools.update_wrapper(generated, func)
    return generated


def compile_dynamic_cell(
    source: str,
    df_name: str,
    columns: list[str],
    namespace: dict[str, Any],
    *,
    filename: str = "<dfmacro-magic>",
) -> tuple[types.CodeType, dict[str, Any], str]:
    """Compile a magic cell with bare column names rewritten to lazy expressions."""
    tree = ast.parse(source, filename=filename)
    columns_name = _unique_name("__dfmacro_columns__", set(namespace))
    collector = _LocalNameCollector()
    for statement in tree.body:
        collector.visit(statement)
    local_names = collector.names - collector.global_names
    protected_names = set(namespace) | set(_BUILTIN_NAMES)
    transformed = _CellNameRewriter(
        columns_name,
        df_name,
        set(columns),
        local_names,
        protected_names,
    ).visit(tree)
    ast.fix_missing_locations(transformed)
    code = compile(transformed, filename, "exec")
    exec_namespace = dict(namespace)
    exec_namespace[columns_name] = _ColumnNamespace()
    return code, exec_namespace, columns_name
