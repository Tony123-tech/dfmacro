"""Registry-driven compilation for generated accessor expression nodes."""

from __future__ import annotations

from typing import Any, Callable, Literal

from .._generated_methods import ACCESSOR_METHODS
from ..expr import AccessorExpr
from . import CompileError


def accessor_capabilities(namespace: str, method: str) -> dict[str, bool]:
    """Return backend support flags for one registered accessor method."""
    spec = ACCESSOR_METHODS.get((namespace, method))
    if spec is None:
        return {"pandas": False, "polars": False}
    return {backend: spec[backend] is not None for backend in ("pandas", "polars")}


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        current = getattr(current, component)
    return current


_CALL_ARGUMENTS = {
    ("str", "contains"): ("pat",),
    ("str", "startswith"): ("pat",),
    ("str", "endswith"): ("pat",),
    ("str", "strip"): ("chars",),
    ("str", "lstrip"): ("chars",),
    ("str", "rstrip"): ("chars",),
    ("str", "replace"): ("pat", "repl"),
    ("str", "split"): ("pat",),
    ("str", "slice"): ("start", "length"),
    ("str", "pad"): ("width",),
    ("str", "zfill"): ("width",),
    ("str", "match"): ("pat",),
    ("str", "extract"): ("pat",),
    ("str", "count"): ("pat",),
    ("str", "fullmatch"): ("pat",),
    ("str", "get"): ("i",),
    ("str", "head"): ("n",),
    ("str", "tail"): ("n",),
    ("str", "repeat"): ("repeats",),
    ("dt", "strftime"): ("fmt",),
    ("dt", "floor"): ("freq",),
    ("dt", "ceil"): ("freq",),
    ("dt", "round"): ("freq",),
    ("dt", "as_unit"): ("unit",),
    ("dt", "to_period"): ("freq",),
    ("dt", "tz_localize"): ("tz",),
    ("dt", "tz_convert"): ("tz",),
}


def _arguments(expr: AccessorExpr) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args = list(expr.args)
    kwargs = dict(expr.kwargs)
    for name in _CALL_ARGUMENTS.get((expr.namespace, expr.method_name), ()):
        if name in kwargs:
            args.append(kwargs.pop(name))
    return tuple(args), kwargs


def _pandas_slice(column: Any, expr: AccessorExpr) -> Any:
    start = expr.kwargs.get("start")
    length = expr.kwargs.get("length")
    stop = None if length is None else (start or 0) + length
    return getattr(column.str, "slice")(start=start, stop=stop)


def _pandas_extract(column: Any, expr: AccessorExpr) -> Any:
    result = getattr(column.str, "extract")(
        expr.kwargs["pat"], expand=True
    )
    group = expr.kwargs.get("group", 1)
    if result.shape[1] < group:
        raise CompileError(
            f"String extraction requested capture group {group}, but the pattern "
            f"contains only {result.shape[1]} capture group(s)."
        )
    return result.iloc[:, group - 1]


def _pandas_reverse(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return column.str[::-1]


def _polars_reverse(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.str, "reverse")()


def _pandas_week(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return column.dt.isocalendar().week


def _polars_contains(column: Any, expr: AccessorExpr) -> Any:
    return getattr(column.str, "contains")(
        expr.kwargs["pat"], literal=not expr.kwargs.get("regex", False)
    )


def _polars_replace(column: Any, expr: AccessorExpr) -> Any:
    return getattr(column.str, "replace")(
        expr.kwargs["pat"],
        expr.kwargs["repl"],
        literal=not expr.kwargs.get("regex", False),
        n=-1,
    )


def _polars_pad(column: Any, expr: AccessorExpr) -> Any:
    import polars as pl

    width = expr.kwargs["width"]
    side = expr.kwargs.get("side", "left")
    fillchar = expr.kwargs.get("fillchar", " ")
    if side == "left":
        return getattr(column.str, "pad_start")(width, fill_char=fillchar)
    if side == "right":
        return getattr(column.str, "pad_end")(width, fill_char=fillchar)
    length = getattr(column.str, "len_chars")()
    left_width = (width + length) // 2
    return (
        pl.when(length < width)
        .then(
            getattr(column.str, "pad_start")(left_width, fill_char=fillchar).str.pad_end(
                width, fill_char=fillchar
            )
        )
        .otherwise(column)
    )


def _polars_match(column: Any, expr: AccessorExpr) -> Any:
    return getattr(column.str, "contains")(f"^(?:{expr.kwargs['pat']})")


def _polars_extract(column: Any, expr: AccessorExpr) -> Any:
    return getattr(column.str, "extract")(
        expr.kwargs["pat"], group_index=expr.kwargs.get("group", 1)
    )


def _polars_fullmatch(column: Any, expr: AccessorExpr) -> Any:
    return getattr(column.str, "contains")(f"^(?:{expr.kwargs['pat']})$")


def _polars_isalnum(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.str, "contains")(r"^[\p{L}\p{N}]+$")


def _polars_isalpha(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.str, "contains")(r"^\p{L}+$")


def _polars_isdigit(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.str, "contains")(r"^\p{Nd}+$")


def _polars_islower(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.str, "contains")(r"\p{Ll}") & (
        getattr(column.str, "to_lowercase")() == column
    )


def _polars_isupper(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.str, "contains")(r"\p{Lu}") & (
        getattr(column.str, "to_uppercase")() == column
    )


def _polars_dayofweek(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.dt, "weekday")() - 1


def _polars_nanosecond(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.dt, "nanosecond")() % 1000


def _polars_dayofyear(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.dt, "ordinal_day")()


def _polars_days_in_month(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.dt, "month_end")().dt.day()


def _polars_is_month_start(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.dt, "day")() == 1


def _polars_is_month_end(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return column == getattr(column.dt, "month_end")()


def _polars_is_year_start(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return (getattr(column.dt, "month")() == 1) & (getattr(column.dt, "day")() == 1)


def _polars_is_year_end(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return (getattr(column.dt, "month")() == 12) & (getattr(column.dt, "day")() == 31)


def _polars_week(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.dt, "week")()


def _polars_name(column: Any, expr: AccessorExpr) -> Any:
    fmt = "%B" if expr.method_name == "month_name" else "%A"
    return getattr(column.dt, "strftime")(fmt)


def _polars_normalize(column: Any, expr: AccessorExpr) -> Any:
    del expr
    return getattr(column.dt, "truncate")("1d")


def _polars_ceil(column: Any, expr: AccessorExpr) -> Any:
    import polars as pl

    truncated = getattr(column.dt, "truncate")(expr.kwargs["freq"])
    return pl.when(column == truncated).then(truncated).otherwise(
        truncated.dt.offset_by(expr.kwargs["freq"])
    )


def _polars_date_floor(column: Any, expr: AccessorExpr) -> Any:
    return getattr(column.dt, "truncate")(expr.kwargs["freq"])


def _get_accessor(column: Any, namespace: str, backend: str) -> Any:
    try:
        return getattr(column, namespace)
    except AttributeError as exc:
        raise CompileError(
            f"Accessor {namespace!r} is not available for the {backend} backend "
            f"for column type {type(column).__name__}."
        ) from exc


_SPECIAL_HANDLERS: dict[
    tuple[str, str, str], Callable[[Any, AccessorExpr], Any]
] = {
    ("pandas", "str", "slice"): _pandas_slice,
    ("pandas", "str", "extract"): _pandas_extract,
    ("pandas", "str", "reverse"): _pandas_reverse,
    ("polars", "str", "reverse"): _polars_reverse,
    ("pandas", "dt", "week"): _pandas_week,
    ("polars", "str", "contains"): _polars_contains,
    ("polars", "str", "replace"): _polars_replace,
    ("polars", "str", "pad"): _polars_pad,
    ("polars", "str", "match"): _polars_match,
    ("polars", "str", "extract"): _polars_extract,
    ("polars", "str", "fullmatch"): _polars_fullmatch,
    ("polars", "str", "isalnum"): _polars_isalnum,
    ("polars", "str", "isalpha"): _polars_isalpha,
    ("polars", "str", "isdigit"): _polars_isdigit,
    ("polars", "str", "islower"): _polars_islower,
    ("polars", "str", "isupper"): _polars_isupper,
    ("polars", "dt", "dayofweek"): _polars_dayofweek,
    ("polars", "dt", "nanosecond"): _polars_nanosecond,
    ("polars", "dt", "dayofyear"): _polars_dayofyear,
    ("polars", "dt", "week"): _polars_week,
    ("polars", "dt", "days_in_month"): _polars_days_in_month,
    ("polars", "dt", "is_month_start"): _polars_is_month_start,
    ("polars", "dt", "is_month_end"): _polars_is_month_end,
    ("polars", "dt", "is_year_start"): _polars_is_year_start,
    ("polars", "dt", "is_year_end"): _polars_is_year_end,
    ("polars", "dt", "month_name"): _polars_name,
    ("polars", "dt", "day_name"): _polars_name,
    ("polars", "dt", "normalize"): _polars_normalize,
    ("polars", "dt", "floor"): _polars_date_floor,
    ("polars", "dt", "ceil"): _polars_ceil,
}


def compile_accessor_method(
    expr: AccessorExpr, column: Any, backend: Literal["pandas", "polars"]
) -> Any:
    from ..plugins import accessor_compiler

    plugin = accessor_compiler(expr.namespace, expr.method_name)
    if plugin is not None:
        try:
            return plugin(column, *expr.args, **expr.kwargs)
        except Exception as exc:
            raise CompileError(
                f"Custom accessor {expr.namespace}.{expr.method_name} failed "
                f"for the {backend} backend: {exc}"
            ) from exc

    handler = _SPECIAL_HANDLERS.get((backend, expr.namespace, expr.method_name))
    if handler is not None:
        try:
            return handler(column, expr)
        except AttributeError as exc:
            raise CompileError(
                f"{expr.namespace}.{expr.method_name} is unavailable on the "
                f"{backend} backend."
            ) from exc

    spec = ACCESSOR_METHODS.get((expr.namespace, expr.method_name))
    if spec is None:
        raise CompileError(
            f"Unknown accessor method {expr.namespace}.{expr.method_name}."
        )

    path = spec[backend]
    if path is None:
        available = ", ".join(
            name for name in ("pandas", "polars") if spec[name] is not None
        )
        raise CompileError(
            f"{expr.namespace}.{expr.method_name} is not supported by the {backend} "
            f"backend. Registered backend support: {available or 'none'}."
        )

    try:
        if "." in path:
            accessor_name, method_path = path.split(".", 1)
            accessor = _get_accessor(column, accessor_name, backend)
            operation = _resolve_path(accessor, method_path)
        else:
            operation = _resolve_path(column, path)
        args, kwargs = _arguments(expr)
        if callable(operation):
            return operation(*args, **kwargs)
        if args or kwargs:
            raise CompileError(
                f"{expr.namespace}.{expr.method_name} is a property and does not "
                "accept arguments."
            )
        return operation
    except AttributeError as exc:
        raise CompileError(
            f"{expr.namespace}.{expr.method_name} is registered for {backend}, but "
            f"the installed backend does not provide {path!r}."
        ) from exc
