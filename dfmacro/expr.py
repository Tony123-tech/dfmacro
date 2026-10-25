"""Lazy expression tree for dfmacro."""

from __future__ import annotations

from typing import Any, cast


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


class StringExpr:
    __slots__ = ("_expr",)

    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    def contains(self, pat: str, regex: bool = False) -> Expr:
        return StringMethodExpr("contains", self._expr, {"pat": pat, "regex": regex})

    def startswith(self, pat: str) -> Expr:
        return StringMethodExpr("startswith", self._expr, {"pat": pat})

    def endswith(self, pat: str) -> Expr:
        return StringMethodExpr("endswith", self._expr, {"pat": pat})

    def lower(self) -> Expr:
        return StringMethodExpr("lower", self._expr, {})

    def upper(self) -> Expr:
        return StringMethodExpr("upper", self._expr, {})

    def strip(self) -> Expr:
        return StringMethodExpr("strip", self._expr, {})

    def len(self) -> Expr:
        return StringMethodExpr("len", self._expr, {})

    def replace(self, pat: str, repl: str) -> Expr:
        return StringMethodExpr("replace", self._expr, {"pat": pat, "repl": repl})

    def split(self, pat: str) -> Expr:
        return StringMethodExpr("split", self._expr, {"pat": pat})

    def slice(self, start: int, length: int) -> Expr:
        return StringMethodExpr("slice", self._expr, {"start": start, "length": length})

    def pad(self, width: int, side: str = "left") -> Expr:
        return StringMethodExpr("pad", self._expr, {"width": width, "side": side})

    def zfill(self, width: int) -> Expr:
        return StringMethodExpr("zfill", self._expr, {"width": width})

    def match(self, pat: str) -> Expr:
        return StringMethodExpr("match", self._expr, {"pat": pat})

    def extract(self, pat: str) -> Expr:
        return StringMethodExpr("extract", self._expr, {"pat": pat})


class StringMethodExpr(Expr):
    __slots__ = ("column", "kwargs", "method")

    def __init__(self, method: str, column: Expr, kwargs: dict) -> None:
        self.method = method
        self.column = column
        self.kwargs = kwargs

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.column!r}.str.{self.method}({args})"


class DateExpr:
    __slots__ = ("_expr",)

    def __init__(self, expr: Expr) -> None:
        self._expr = expr

    @property
    def year(self) -> Expr:
        return DatePartExpr("year", self._expr)

    @property
    def month(self) -> Expr:
        return DatePartExpr("month", self._expr)

    @property
    def day(self) -> Expr:
        return DatePartExpr("day", self._expr)

    @property
    def hour(self) -> Expr:
        return DatePartExpr("hour", self._expr)

    @property
    def minute(self) -> Expr:
        return DatePartExpr("minute", self._expr)

    @property
    def dayofweek(self) -> Expr:
        return DatePartExpr("dayofweek", self._expr)

    @property
    def quarter(self) -> Expr:
        return DatePartExpr("quarter", self._expr)

    def strftime(self, fmt: str) -> Expr:
        return DatePartExpr("strftime", self._expr, {"fmt": fmt})


class DatePartExpr(Expr):
    __slots__ = ("column", "kwargs", "part")

    def __init__(self, part: str, column: Expr, kwargs: dict | None = None) -> None:
        self.part = part
        self.column = column
        self.kwargs = kwargs or {}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.column!r}.dt.{self.part}({args})"


class WhenExpr(Expr):
    __slots__ = ("condition", "then_value", "else_value")

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
