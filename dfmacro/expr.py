"""Lazy expression tree for dfmacro."""

from __future__ import annotations

from typing import Any, cast

from ._generated_methods import ACCESSOR_METHODS


class Expr:
    __slots__ = ()

    def __bool__(self) -> bool:
        raise TypeError(
            "ColumnExpr cannot be used with 'and'/'or'. Use '&'/'|' instead."
        )

    # Arithmetic
    def __add__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("+", self, _wrap(other))

    def __radd__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("+", _wrap(other), self)

    def __sub__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("-", self, _wrap(other))

    def __rsub__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("-", _wrap(other), self)

    def __mul__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("*", self, _wrap(other))

    def __rmul__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("*", _wrap(other), self)

    def __truediv__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("/", self, _wrap(other))

    def __rtruediv__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("/", _wrap(other), self)

    # Comparisons
    def __gt__(self, other: Any) -> BinaryExpr:
        _check_precedence_trap(">", other)
        return BinaryExpr(">", self, _wrap(other))

    def __ge__(self, other: Any) -> BinaryExpr:
        _check_precedence_trap(">=", other)
        return BinaryExpr(">=", self, _wrap(other))

    def __lt__(self, other: Any) -> BinaryExpr:
        _check_precedence_trap("<", other)
        return BinaryExpr("<", self, _wrap(other))

    def __le__(self, other: Any) -> BinaryExpr:
        _check_precedence_trap("<=", other)
        return BinaryExpr("<=", self, _wrap(other))

    def __eq__(self, other: object) -> BinaryExpr:  # type: ignore[override]
        return BinaryExpr("==", self, _wrap(other))

    def __ne__(self, other: object) -> BinaryExpr:  # type: ignore[override]
        return BinaryExpr("!=", self, _wrap(other))

    # Boolean
    def __and__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("&", self, _wrap(other))

    def __rand__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("&", _wrap(other), self)

    def __or__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("|", self, _wrap(other))

    def __ror__(self, other: Any) -> BinaryExpr:
        return BinaryExpr("|", _wrap(other), self)

    def __invert__(self) -> UnaryExpr:
        return UnaryExpr("~", self)

    # Aggregations
    def mean(self) -> AggExpr:
        return AggExpr("mean", self)

    def sum(self) -> AggExpr:
        return AggExpr("sum", self)

    def min(self) -> AggExpr:
        return AggExpr("min", self)

    def max(self) -> AggExpr:
        return AggExpr("max", self)

    def count(self) -> AggExpr:
        return AggExpr("count", self)

    def median(self) -> AggExpr:
        return AggExpr("median", self)

    def std(self) -> AggExpr:
        return AggExpr("std", self)

    def var(self) -> AggExpr:
        return AggExpr("var", self)

    def first(self) -> AggExpr:
        return AggExpr("first", self)

    def last(self) -> AggExpr:
        return AggExpr("last", self)

    def nunique(self) -> AggExpr:
        return AggExpr("nunique", self)

    def quantile(self, q: float) -> AggExpr:
        return AggExpr("quantile", self, {"q": q})

    # Window Functions
    def rank(self, method: str = "average", descending: bool = False) -> WindowExpr:
        return WindowExpr("rank", self, {"method": method, "descending": descending})

    def cumsum(self) -> WindowExpr:
        return WindowExpr("cumsum", self, {})

    def cummax(self) -> WindowExpr:
        return WindowExpr("cummax", self, {})

    def cummin(self) -> WindowExpr:
        return WindowExpr("cummin", self, {})

    def shift(self, periods: int = 1) -> WindowExpr:
        return WindowExpr("shift", self, {"periods": periods})

    def diff(self) -> WindowExpr:
        return WindowExpr("diff", self, {})

    def pct_change(self) -> WindowExpr:
        return WindowExpr("pct_change", self, {})

    # Math
    def abs(self) -> MathExpr:
        return MathExpr("abs", self, {})

    def round(self, n: int = 0) -> MathExpr:
        return MathExpr("round", self, {"n": n})

    def sqrt(self) -> MathExpr:
        return MathExpr("sqrt", self, {})

    def pow(self, n: float) -> MathExpr:
        return MathExpr("pow", self, {"n": n})

    # Accessors
    @property
    def str(self) -> StringExpr:
        return StringExpr(self)

    @property
    def dt(self) -> DateExpr:
        return DateExpr(self)

    @property
    def cat(self) -> CatExpr:
        return CatExpr(self)

    @property
    def num(self) -> NumExpr:
        return NumExpr(self)

    @property
    def list(self) -> ListExpr:
        return ListExpr(self)

    @property
    def struct(self) -> StructExpr:
        return StructExpr(self)

def _wrap(value: Any) -> Expr | Literal:
    if isinstance(value, Expr):
        return value
    return Literal(value)


def _check_precedence_trap(op: str, other: Any) -> None:
    if isinstance(other, Expr):
        raise TypeError(
            f"Operator precedence trap detected: '{op}' received an expression as its right operand. Wrap the comparison in parentheses."
        )


class ColumnExpr(Expr):
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"col({self.name!r})"


class Literal(Expr):
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"lit({self.value!r})"


class BinaryExpr(Expr):
    __slots__ = ("left", "op", "right")

    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


class UnaryExpr(Expr):
    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand: Expr) -> None:
        self.op = op
        self.operand = operand

    def __repr__(self) -> str:
        return f"({self.op}{self.operand!r})"


class AggExpr(Expr):
    __slots__ = ("column", "func", "kwargs")

    def __init__(self, func: str, column: Expr, kwargs: dict | None = None) -> None:
        self.func = func
        self.column = column
        self.kwargs = kwargs or {}

    def __repr__(self) -> str:
        return f"{self.func}({self.column!r})"


class WindowExpr(Expr):
    __slots__ = ("column", "func", "kwargs")

    def __init__(self, func: str, column: Expr, kwargs: dict) -> None:
        self.func = func
        self.column = column
        self.kwargs = kwargs

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.column!r}.{self.func}({args})"


class MathExpr(Expr):
    __slots__ = ("column", "func", "kwargs")

    def __init__(self, func: str, column: Expr, kwargs: dict) -> None:
        self.func = func
        self.column = column
        self.kwargs = kwargs

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.column!r}.{self.func}({args})"


class AccessorExpr(Expr):
    __slots__ = ("args", "column", "is_property", "kwargs", "method_name", "namespace")

    def __init__(
        self,
        namespace: str,
        method_name: str,
        column: Expr,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        is_property: bool = False,
    ) -> None:
        self.namespace = namespace
        self.method_name = method_name
        self.column = column
        self.args = args
        self.kwargs = kwargs or {}
        self.is_property = is_property

    def __repr__(self) -> str:
        if self.is_property:
            return f"{self.column!r}.{self.namespace}.{self.method_name}"
        args = [repr(value) for value in self.args]
        args.extend(f"{key}={value!r}" for key, value in self.kwargs.items())
        return (
            f"{self.column!r}.{self.namespace}.{self.method_name}"
            f"({', '.join(args)})"
        )


class StringExpr:
    __slots__ = ("_expr",)

    _expr: Expr

    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def contains(self, pat: str, regex: bool = False) -> StringMethodExpr:
        return StringMethodExpr("contains", self._expr, {"pat": pat, "regex": regex})

    def startswith(self, pat: str) -> StringMethodExpr:
        return StringMethodExpr("startswith", self._expr, {"pat": pat})

    def endswith(self, pat: str) -> StringMethodExpr:
        return StringMethodExpr("endswith", self._expr, {"pat": pat})

    def lower(self) -> StringMethodExpr:
        return StringMethodExpr("lower", self._expr, {})

    def upper(self) -> StringMethodExpr:
        return StringMethodExpr("upper", self._expr, {})

    def strip(self, chars: str | None = None) -> StringMethodExpr:
        return StringMethodExpr("strip", self._expr, {"chars": chars})

    def len(self) -> StringMethodExpr:
        return StringMethodExpr("len", self._expr, {})

    def replace(
        self, pat: str, repl: str, regex: bool = False
    ) -> StringMethodExpr:
        return StringMethodExpr(
            "replace", self._expr, {"pat": pat, "repl": repl, "regex": regex}
        )

    def split(self, pat: str) -> StringMethodExpr:
        return StringMethodExpr("split", self._expr, {"pat": pat})

    def slice(
        self, start: int | None = None, length: int | None = None
    ) -> StringMethodExpr:
        return StringMethodExpr(
            "slice", self._expr, {"start": start, "length": length}
        )

    def pad(
        self, width: int, side: str = "left", fillchar: str = " "
    ) -> StringMethodExpr:
        if side not in {"left", "right", "both"}:
            raise ValueError("side must be 'left', 'right', or 'both'.")
        if len(fillchar) != 1:
            raise ValueError("fillchar must be exactly one character.")
        return StringMethodExpr(
            "pad",
            self._expr,
            {"width": width, "side": side, "fillchar": fillchar},
        )

    def zfill(self, width: int) -> StringMethodExpr:
        return StringMethodExpr("zfill", self._expr, {"width": width})

    def match(self, pat: str) -> StringMethodExpr:
        return StringMethodExpr("match", self._expr, {"pat": pat})

    def extract(self, pat: str, group: int = 1) -> StringMethodExpr:
        if group < 1:
            raise ValueError("group must be a positive, 1-based capture index.")
        return StringMethodExpr("extract", self._expr, {"pat": pat, "group": group})

    def lstrip(self, chars: str | None = None) -> StringMethodExpr:
        return StringMethodExpr("lstrip", self._expr, {"chars": chars})

    def rstrip(self, chars: str | None = None) -> StringMethodExpr:
        return StringMethodExpr("rstrip", self._expr, {"chars": chars})

    def title(self) -> StringMethodExpr:
        return StringMethodExpr("title", self._expr, {})

    def count(self, pat: str) -> StringMethodExpr:
        return StringMethodExpr("count", self._expr, {"pat": pat})

    def reverse(self) -> StringMethodExpr:
        return StringMethodExpr("reverse", self._expr, {})

    def fullmatch(self, pat: str) -> StringMethodExpr:
        return StringMethodExpr("fullmatch", self._expr, {"pat": pat})

    def isalnum(self) -> StringMethodExpr:
        return StringMethodExpr("isalnum", self._expr, {})

    def isalpha(self) -> StringMethodExpr:
        return StringMethodExpr("isalpha", self._expr, {})

    def isdigit(self) -> StringMethodExpr:
        return StringMethodExpr("isdigit", self._expr, {})

    def islower(self) -> StringMethodExpr:
        return StringMethodExpr("islower", self._expr, {})

    def isupper(self) -> StringMethodExpr:
        return StringMethodExpr("isupper", self._expr, {})


class StringMethodExpr(AccessorExpr):
    __slots__ = ()

    def __init__(
        self,
        method: str,
        column: Expr,
        kwargs: dict[str, Any],
        args: tuple[Any, ...] = (),
    ) -> None:
        super().__init__("str", method, column, args=args, kwargs=kwargs)

    @property
    def method(self) -> str:
        return self.method_name

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.column!r}.str.{self.method}({args})"


class DateExpr:
    __slots__ = ("_expr",)

    _expr: Expr

    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    @property
    def year(self) -> DatePartExpr:
        return DatePartExpr("year", self._expr)

    @property
    def month(self) -> DatePartExpr:
        return DatePartExpr("month", self._expr)

    @property
    def day(self) -> DatePartExpr:
        return DatePartExpr("day", self._expr)

    @property
    def hour(self) -> DatePartExpr:
        return DatePartExpr("hour", self._expr)

    @property
    def minute(self) -> DatePartExpr:
        return DatePartExpr("minute", self._expr)

    @property
    def second(self) -> DatePartExpr:
        return DatePartExpr("second", self._expr)

    @property
    def microsecond(self) -> DatePartExpr:
        return DatePartExpr("microsecond", self._expr)

    @property
    def nanosecond(self) -> DatePartExpr:
        return DatePartExpr("nanosecond", self._expr)

    @property
    def dayofweek(self) -> DatePartExpr:
        return DatePartExpr("dayofweek", self._expr)

    @property
    def quarter(self) -> DatePartExpr:
        return DatePartExpr("quarter", self._expr)

    @property
    def dayofyear(self) -> DatePartExpr:
        return DatePartExpr("dayofyear", self._expr)

    @property
    def week(self) -> DatePartExpr:
        return DatePartExpr("week", self._expr)

    @property
    def days_in_month(self) -> DatePartExpr:
        return DatePartExpr("days_in_month", self._expr)

    @property
    def is_month_start(self) -> DatePartExpr:
        return DatePartExpr("is_month_start", self._expr)

    @property
    def is_month_end(self) -> DatePartExpr:
        return DatePartExpr("is_month_end", self._expr)

    @property
    def is_year_start(self) -> DatePartExpr:
        return DatePartExpr("is_year_start", self._expr)

    @property
    def is_year_end(self) -> DatePartExpr:
        return DatePartExpr("is_year_end", self._expr)

    @property
    def is_leap_year(self) -> DatePartExpr:
        return DatePartExpr("is_leap_year", self._expr)

    @property
    def month_name(self) -> DatePartExpr:
        return DatePartExpr("month_name", self._expr)

    @property
    def day_name(self) -> DatePartExpr:
        return DatePartExpr("day_name", self._expr)

    def normalize(self) -> DatePartExpr:
        return DatePartExpr("normalize", self._expr)

    def floor(self, freq: str) -> DatePartExpr:
        return DatePartExpr("floor", self._expr, {"freq": freq})

    def ceil(self, freq: str) -> DatePartExpr:
        return DatePartExpr("ceil", self._expr, {"freq": freq})

    def round(self, freq: str) -> DatePartExpr:
        return DatePartExpr("round", self._expr, {"freq": freq})

    def strftime(self, fmt: str) -> DatePartExpr:
        return DatePartExpr("strftime", self._expr, {"fmt": fmt})


class DatePartExpr(AccessorExpr):
    __slots__ = ()

    def __init__(
        self,
        part: str,
        column: Expr,
        kwargs: dict[str, Any] | None = None,
        is_property: bool | None = None,
    ) -> None:
        super().__init__(
            "dt",
            part,
            column,
            kwargs=kwargs,
            is_property=(not bool(kwargs)) if is_property is None else is_property,
        )

    @property
    def part(self) -> str:
        return self.method_name

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.column!r}.dt.{self.part}({args})"


AccessorMethodExpr = AccessorExpr


class CatExpr:
    __slots__ = ("_expr",)

    def __init__(self, expr: Expr) -> None:
        self._expr = expr


class NumExpr:
    __slots__ = ("_expr",)

    def __init__(self, expr: Expr) -> None:
        self._expr = expr


class ListExpr:
    __slots__ = ("_expr",)

    def __init__(self, expr: Expr) -> None:
        self._expr = expr


class StructExpr:
    __slots__ = ("_expr",)

    def __init__(self, expr: Expr) -> None:
        self._expr = expr


class WhenExpr(Expr):
    __slots__ = ("condition", "else_value", "then_value")

    def __init__(
        self, condition: Expr, then_value: Expr, else_value: Expr | None = None
    ) -> None:
        self.condition = condition
        self.then_value = then_value
        self.else_value = else_value

    def otherwise(self, else_value: Any) -> Expr:
        return WhenExpr(self.condition, self.then_value, _wrap(else_value))

    def __repr__(self) -> str:
        if self.else_value is not None:
            return f"when({self.condition!r}, {self.then_value!r}).otherwise({self.else_value!r})"
        return f"when({self.condition!r}, {self.then_value!r})"


def when(condition: Expr, then_value: Any) -> WhenExpr:
    return WhenExpr(condition, _wrap(then_value))


def structural_eq(a: object, b: object) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, ColumnExpr):
        return cast(ColumnExpr, a).name == cast(ColumnExpr, b).name
    if isinstance(a, Literal):
        return cast(Literal, a).value == cast(Literal, b).value
    if isinstance(a, BinaryExpr):
        return (
            cast(BinaryExpr, a).op == cast(BinaryExpr, b).op
            and structural_eq(cast(BinaryExpr, a).left, cast(BinaryExpr, b).left)
            and structural_eq(cast(BinaryExpr, a).right, cast(BinaryExpr, b).right)
        )
    if isinstance(a, UnaryExpr):
        return cast(UnaryExpr, a).op == cast(UnaryExpr, b).op and structural_eq(
            cast(UnaryExpr, a).operand, cast(UnaryExpr, b).operand
        )
    if isinstance(a, AggExpr):
        return (
            cast(AggExpr, a).func == cast(AggExpr, b).func
            and structural_eq(cast(AggExpr, a).column, cast(AggExpr, b).column)
            and cast(AggExpr, a).kwargs == cast(AggExpr, b).kwargs
        )
    if isinstance(a, WindowExpr):
        return (
            cast(WindowExpr, a).func == cast(WindowExpr, b).func
            and structural_eq(cast(WindowExpr, a).column, cast(WindowExpr, b).column)
            and cast(WindowExpr, a).kwargs == cast(WindowExpr, b).kwargs
        )
    if isinstance(a, MathExpr):
        return (
            cast(MathExpr, a).func == cast(MathExpr, b).func
            and structural_eq(cast(MathExpr, a).column, cast(MathExpr, b).column)
            and cast(MathExpr, a).kwargs == cast(MathExpr, b).kwargs
        )
    if isinstance(a, StringMethodExpr):
        return (
            cast(StringMethodExpr, a).method == cast(StringMethodExpr, b).method
            and structural_eq(
                cast(StringMethodExpr, a).column, cast(StringMethodExpr, b).column
            )
            and cast(StringMethodExpr, a).kwargs == cast(StringMethodExpr, b).kwargs
        )
    if isinstance(a, DatePartExpr):
        return (
            cast(DatePartExpr, a).part == cast(DatePartExpr, b).part
            and structural_eq(
                cast(DatePartExpr, a).column, cast(DatePartExpr, b).column
            )
            and cast(DatePartExpr, a).kwargs == cast(DatePartExpr, b).kwargs
        )
    if isinstance(a, AccessorMethodExpr):
        return (
            cast(AccessorMethodExpr, a).namespace
            == cast(AccessorMethodExpr, b).namespace
            and cast(AccessorMethodExpr, a).method_name
            == cast(AccessorMethodExpr, b).method_name
            and structural_eq(
                cast(AccessorMethodExpr, a).column,
                cast(AccessorMethodExpr, b).column,
            )
            and cast(AccessorMethodExpr, a).args
            == cast(AccessorMethodExpr, b).args
            and cast(AccessorMethodExpr, a).kwargs
            == cast(AccessorMethodExpr, b).kwargs
            and cast(AccessorMethodExpr, a).is_property
            == cast(AccessorMethodExpr, b).is_property
        )
    if isinstance(a, WhenExpr):
        return (
            structural_eq(cast(WhenExpr, a).condition, cast(WhenExpr, b).condition)
            and structural_eq(
                cast(WhenExpr, a).then_value, cast(WhenExpr, b).then_value
            )
            and (
                (
                    cast(WhenExpr, a).else_value is None
                    and cast(WhenExpr, b).else_value is None
                )
                or (
                    cast(WhenExpr, a).else_value is not None
                    and cast(WhenExpr, b).else_value is not None
                    and structural_eq(
                        cast(WhenExpr, a).else_value, cast(WhenExpr, b).else_value
                    )
                )
            )
        )
    return a is b


_ACCESSOR_CLASSES = {
    "str": StringExpr,
    "dt": DateExpr,
    "cat": CatExpr,
    "num": NumExpr,
    "list": ListExpr,
    "struct": StructExpr,
}

_POSITIONAL_ARGUMENTS = {
    "extractall": ("pat",),
    "get": ("i",),
    "repeat": ("repeats",),
    "contains": ("pat", "regex"),
    "startswith": ("pat",),
    "endswith": ("pat",),
    "strip": ("chars",),
    "lstrip": ("chars",),
    "rstrip": ("chars",),
    "replace": ("pat", "repl", "regex"),
    "split": ("pat",),
    "slice": ("start", "length"),
    "pad": ("width", "side", "fillchar"),
    "zfill": ("width",),
    "match": ("pat",),
    "extract": ("pat", "group"),
    "count": ("pat",),
    "fullmatch": ("pat",),
    "strftime": ("fmt",),
    "floor": ("freq",),
    "ceil": ("freq",),
    "round": ("freq",),
    "normalize": (),
    "as_unit": ("unit",),
    "to_period": ("freq",),
    "tz_localize": ("tz",),
    "tz_convert": ("tz",),
}

_ACCESSOR_DEFAULTS = {
    ("str", "contains"): {"regex": False},
    ("str", "strip"): {"chars": None},
    ("str", "lstrip"): {"chars": None},
    ("str", "rstrip"): {"chars": None},
    ("str", "replace"): {"regex": False},
    ("str", "slice"): {"start": None, "length": None},
    ("str", "extract"): {"group": 1},
    ("str", "pad"): {"side": "left", "fillchar": " "},
}


def _make_accessor_expression(
    namespace: str,
    method_name: str,
    column: Expr,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    is_property: bool,
) -> AccessorExpr:
    if is_property and (args or kwargs):
        raise TypeError(
            f"Accessor property {namespace}.{method_name} does not accept arguments."
        )

    if namespace == "str":
        argument_names = _POSITIONAL_ARGUMENTS.get(method_name, ())
        if len(args) > len(argument_names):
            raise TypeError(
                f"{namespace}.{method_name} accepts at most "
                f"{len(argument_names)} positional arguments."
            )
        normalized_kwargs = dict(zip(argument_names, args))
        overlap = normalized_kwargs.keys() & kwargs.keys()
        if overlap:
            duplicated = ", ".join(sorted(overlap))
            raise TypeError(f"Multiple values for argument(s): {duplicated}.")
        normalized_kwargs.update(kwargs)
        for key, value in _ACCESSOR_DEFAULTS.get((namespace, method_name), {}).items():
            normalized_kwargs.setdefault(key, value)
        if method_name == "pad":
            if normalized_kwargs.get("side") not in {"left", "right", "both"}:
                raise ValueError("side must be 'left', 'right', or 'both'.")
            if len(normalized_kwargs.get("fillchar", " ")) != 1:
                raise ValueError("fillchar must be exactly one character.")
        if method_name == "extract" and normalized_kwargs.get("group", 1) < 1:
            raise ValueError("group must be a positive, 1-based capture index.")
        return StringMethodExpr(method_name, column, normalized_kwargs)

    if namespace == "dt":
        argument_names = _POSITIONAL_ARGUMENTS.get(method_name, ())
        if len(args) > len(argument_names):
            raise TypeError(
                f"{namespace}.{method_name} accepts at most "
                f"{len(argument_names)} positional arguments."
            )
        normalized_kwargs = dict(zip(argument_names, args))
        overlap = normalized_kwargs.keys() & kwargs.keys()
        if overlap:
            duplicated = ", ".join(sorted(overlap))
            raise TypeError(f"Multiple values for argument(s): {duplicated}.")
        normalized_kwargs.update(kwargs)
        return DatePartExpr(
            method_name,
            column,
            normalized_kwargs,
            is_property=is_property,
        )

    return AccessorExpr(
        namespace,
        method_name,
        column,
        args=args,
        kwargs=kwargs,
        is_property=is_property,
    )


def _install_accessor_method(
    namespace: str, method_name: str, is_property: bool
) -> None:
    accessor_class = _ACCESSOR_CLASSES[namespace]

    if is_property:
        def get_property(self: Any) -> AccessorExpr:
            return _make_accessor_expression(
                namespace, method_name, self._expr, (), {}, True
            )

        setattr(accessor_class, method_name, property(get_property))
        return

    def accessor_method(self: Any, *args: Any, **kwargs: Any) -> AccessorExpr:
        return _make_accessor_expression(
            namespace, method_name, self._expr, args, kwargs, False
        )

    accessor_method.__name__ = method_name
    accessor_method.__qualname__ = f"{accessor_class.__name__}.{method_name}"
    setattr(accessor_class, method_name, accessor_method)


for (_namespace, _method_name), _method_spec in ACCESSOR_METHODS.items():
    _install_accessor_method(
        _namespace,
        _method_name,
        _method_spec["property"],
    )
