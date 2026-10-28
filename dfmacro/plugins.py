"""Registration hooks for custom accessor methods and backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .expr import (
    AccessorExpr,
    CatExpr,
    DateExpr,
    ListExpr,
    NumExpr,
    StringExpr,
    StructExpr,
)

BackendCompiler = Callable[[Any, Any], Any]
AccessorCompiler = Callable[..., Any]

_BACKENDS: dict[str, BackendCompiler] = {}
_ACCESSORS: dict[tuple[str, str], AccessorCompiler] = {}
_ACCESSOR_CLASSES = {
    "str": StringExpr,
    "dt": DateExpr,
    "cat": CatExpr,
    "num": NumExpr,
    "list": ListExpr,
    "struct": StructExpr,
}


def register_backend(name: str) -> Callable[[BackendCompiler], BackendCompiler]:
    """Register a compiler for DataFrames whose module/type identifies ``name``.

    The compiler receives ``(expr, dataframe)``. A DataFrame may identify its
    plugin by defining ``__dfmacro_backend__ = name`` on its class, or by using
    a module name beginning with ``name``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Backend name must be a non-empty string.")

    def decorate(compiler: BackendCompiler) -> BackendCompiler:
        if name in _BACKENDS:
            raise ValueError(f"A backend named {name!r} is already registered.")
        _BACKENDS[name] = compiler
        return compiler

    return decorate


def register_accessor(name: str) -> Callable[[AccessorCompiler], AccessorCompiler]:
    """Register a method builder/compiler, such as ``"str.my_custom_lower"``.

    The plugin function is called at execution with the native source column
    followed by the expression's positional and keyword arguments.
    """
    parts = name.split(".", 1) if isinstance(name, str) else []
    if len(parts) != 2 or parts[0] not in _ACCESSOR_CLASSES or not parts[1]:
        raise ValueError(
            "Accessor names must be '<namespace>.<method>' using a supported namespace."
        )
    namespace, method = parts
    key = (namespace, method)

    def decorate(function: AccessorCompiler) -> AccessorCompiler:
        if key in _ACCESSORS:
            raise ValueError(f"Accessor {name!r} is already registered.")
        _ACCESSORS[key] = function
        accessor_class = _ACCESSOR_CLASSES[namespace]

        def accessor_method(self: Any, *args: Any, **kwargs: Any) -> AccessorExpr:
            return AccessorExpr(
                namespace,
                method,
                self._expr,
                args=args,
                kwargs=kwargs,
            )

        setattr(accessor_class, method, accessor_method)
        return function

    return decorate


def backend_compiler_for(dataframe: Any) -> BackendCompiler | None:
    """Find a registered backend compiler for a DataFrame instance."""
    declared = getattr(type(dataframe), "__dfmacro_backend__", None)
    if isinstance(declared, str) and declared in _BACKENDS:
        return _BACKENDS[declared]
    module = type(dataframe).__module__
    for name, compiler in _BACKENDS.items():
        if module == name or module.startswith(name + "."):
            return compiler
    return None


def accessor_compiler(namespace: str, method: str) -> AccessorCompiler | None:
    """Return a registered accessor implementation, if one exists."""
    return _ACCESSORS.get((namespace, method))
