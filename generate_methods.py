#!/usr/bin/env python3
"""Generate the accessor capability registry used by dfmacro expression compilers.

Run from the repository root:

    python generate_methods.py

The method list is intentionally declarative. Backend paths are API paths from a
Series/Expr object; ``None`` means that backend does not advertise the method.
Methods whose backend semantics require normalization should continue using the
existing specialized expression node and compiler implementation.
"""

from __future__ import annotations

import pprint
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dfmacro" / "_generated_methods.py"
PathSpec = tuple[str, Optional[str], Optional[str], bool]

# Names are grouped by public accessor. The two paths are validated at runtime
# because backend APIs differ by version and optional dependencies.
METHODS: dict[str, list[PathSpec]] = {
    "str": [
        ("contains", "str.contains", "str.contains", False),
        ("startswith", "str.startswith", "str.starts_with", False),
        ("endswith", "str.endswith", "str.ends_with", False),
        ("lower", "str.lower", "str.to_lowercase", False),
        ("upper", "str.upper", "str.to_uppercase", False),
        ("strip", "str.strip", "str.strip_chars", False),
        ("lstrip", "str.lstrip", "str.strip_chars_start", False),
        ("rstrip", "str.rstrip", "str.strip_chars_end", False),
        ("len", "str.len", "str.len_chars", False),
        ("replace", "str.replace", "str.replace", False),
        ("split", "str.split", "str.split", False),
        ("slice", "str.slice", "str.slice", False),
        ("pad", "str.pad", None, False),
        ("zfill", "str.zfill", "str.zfill", False),
        ("match", "str.match", None, False),
        ("extract", "str.extract", "str.extract", False),
        ("title", "str.title", "str.to_titlecase", False),
        ("count", "str.count", "str.count_matches", False),
        ("fullmatch", "str.fullmatch", None, False),
        ("isalnum", "str.isalnum", None, False),
        ("isalpha", "str.isalpha", None, False),
        ("isdigit", "str.isdigit", None, False),
        ("islower", "str.islower", None, False),
        ("isupper", "str.isupper", None, False),
        ("capitalize", "str.capitalize", "str.to_titlecase", False),
        ("casefold", "str.casefold", "str.to_lowercase", False),
        ("center", "str.center", "str.pad_start", False),
        ("ljust", "str.ljust", "str.pad_start", False),
        ("rjust", "str.rjust", "str.pad_end", False),
        ("find", "str.find", "str.find", False),
        ("rfind", "str.rfind", "str.rfind", False),
        ("index", "str.index", None, False),
        ("rindex", "str.rindex", None, False),
        ("join", "str.join", "str.join", False),
        ("partition", "str.partition", "str.split_exact", False),
        ("rpartition", "str.rpartition", None, False),
        ("removeprefix", "str.removeprefix", "str.strip_prefix", False),
        ("removesuffix", "str.removesuffix", "str.strip_suffix", False),
        ("wrap", "str.wrap", None, False),
        ("get", "str.get", None, False),
        ("normalize", "str.normalize", "str.normalize", False),
        ("encode", "str.encode", None, False),
        ("decode", "str.decode", None, False),
        ("extractall", "str.extractall", None, False),
        ("get_dummies", "str.get_dummies", None, False),
        ("translate", "str.translate", None, False),
        ("repeat", "str.repeat", None, False),
        ("contains_any", None, "str.contains_any", False),
        ("contains_all", None, "str.contains_all", False),
        ("head", "str.slice", "str.head", False),
        ("tail", "str.slice", "str.tail", False),
    ],
    "dt": [
        ("year", "dt.year", "dt.year", True),
        ("month", "dt.month", "dt.month", True),
        ("day", "dt.day", "dt.day", True),
        ("hour", "dt.hour", "dt.hour", True),
        ("minute", "dt.minute", "dt.minute", True),
        ("second", "dt.second", "dt.second", True),
        ("microsecond", "dt.microsecond", "dt.microsecond", True),
        ("nanosecond", "dt.nanosecond", "dt.nanosecond", True),
        ("dayofweek", "dt.dayofweek", "dt.weekday", True),
        ("weekday", "dt.weekday", "dt.weekday", True),
        ("quarter", "dt.quarter", "dt.quarter", True),
        ("dayofyear", "dt.dayofyear", "dt.ordinal_day", True),
        ("week", "dt.isocalendar", "dt.week", True),
        ("days_in_month", "dt.days_in_month", None, True),
        ("is_month_start", "dt.is_month_start", None, True),
        ("is_month_end", "dt.is_month_end", None, True),
        ("is_year_start", "dt.is_year_start", None, True),
        ("is_year_end", "dt.is_year_end", None, True),
        ("is_leap_year", "dt.is_leap_year", "dt.is_leap_year", True),
        ("month_name", "dt.month_name", "dt.strftime", True),
        ("day_name", "dt.day_name", "dt.strftime", True),
        ("date", "dt.date", "dt.date", True),
        ("time", "dt.time", None, False),
        ("timetz", "dt.timetz", None, False),
        ("tz", "dt.tz", "dt.time_zone", True),
        ("unit", "dt.unit", None, True),
        ("normalize", "dt.normalize", "dt.truncate", False),
        ("floor", "dt.floor", "dt.truncate", False),
        ("ceil", "dt.ceil", None, False),
        ("round", "dt.round", "dt.round", False),
        ("strftime", "dt.strftime", "dt.strftime", False),
        ("as_unit", "dt.as_unit", "dt.cast_time_unit", False),
        ("to_period", "dt.to_period", None, False),
        ("to_pydatetime", "dt.to_pydatetime", None, False),
        ("isocalendar", "dt.isocalendar", None, False),
        ("total_seconds", "dt.total_seconds", "dt.total_seconds", False),
        ("components", "dt.components", None, False),
        ("days", "dt.days", "dt.total_days", True),
        ("seconds", "dt.seconds", "dt.total_seconds", True),
        ("milliseconds", "dt.milliseconds", "dt.total_milliseconds", True),
        ("microseconds", "dt.microseconds", "dt.total_microseconds", True),
        ("nanoseconds", "dt.nanoseconds", "dt.total_nanoseconds", True),
        ("to_pytimedelta", "dt.to_pytimedelta", None, False),
        ("tz_localize", "dt.tz_localize", "dt.replace_time_zone", False),
        ("tz_convert", "dt.tz_convert", "dt.convert_time_zone", False),
    ],
    "cat": [
        ("categories", "cat.categories", "cat.get_categories", True),
        ("codes", "cat.codes", "to_physical", True),
        ("ordered", "cat.ordered", None, True),
        ("set_categories", "cat.set_categories", "cat.set_categories", False),
        ("rename_categories", "cat.rename_categories", "cat.rename_categories", False),
        (
            "reorder_categories",
            "cat.reorder_categories",
            "cat.reorder_categories",
            False,
        ),
        ("add_categories", "cat.add_categories", "cat.add_categories", False),
        ("remove_categories", "cat.remove_categories", "cat.remove_categories", False),
        (
            "remove_unused_categories",
            "cat.remove_unused_categories",
            "cat.remove_unused_categories",
            False,
        ),
        ("as_ordered", "cat.as_ordered", "cat.set_ordering", False),
        ("as_unordered", "cat.as_unordered", "cat.set_ordering", False),
        ("get_categories", "cat.categories", "cat.get_categories", True),
    ],
    "num": [
        ("abs", "abs", "abs", False),
        ("round", "round", "round", False),
        ("sqrt", None, "sqrt", False),
        ("log", None, "log", False),
        ("exp", None, "exp", False),
        ("sign", None, "sign", False),
        ("ceil", None, "ceil", False),
        ("floor", None, "floor", False),
        ("clip", "clip", "clip", False),
        ("pow", "pow", "pow", False),
        ("add", "add", "add", False),
        ("sub", "sub", "sub", False),
        ("mul", "mul", "mul", False),
        ("div", "div", "truediv", False),
        ("mod", "mod", "mod", False),
        ("floordiv", "floordiv", "floordiv", False),
        ("gt", "gt", "gt", False),
        ("ge", "ge", "ge", False),
        ("lt", "lt", "lt", False),
        ("le", "le", "le", False),
        ("eq", "eq", "eq", False),
        ("ne", "ne", "ne", False),
        ("is_finite", None, "is_finite", False),
        ("is_infinite", None, "is_infinite", False),
        ("is_nan", "isna", "is_nan", False),
        ("is_null", "isna", "is_null", False),
        ("fill_null", "fillna", "fill_null", False),
        ("round_sig_figs", None, None, False),
        ("log10", None, "log10", False),
        ("log1p", None, "log1p", False),
        ("exp2", None, "exp2", False),
        ("arcsin", None, "arcsin", False),
        ("arccos", None, "arccos", False),
        ("arctan", None, "arctan", False),
        ("sinh", None, "sinh", False),
        ("cosh", None, "cosh", False),
        ("tanh", None, "tanh", False),
        ("degrees", None, "degrees", False),
        ("radians", None, "radians", False),
    ],
    "list": [
        ("len", None, "list.len", False),
        ("get", None, "list.get", False),
        ("first", None, "list.first", False),
        ("last", None, "list.last", False),
        ("join", None, "list.join", False),
        ("contains", None, "list.contains", False),
        ("sort", None, "list.sort", False),
        ("reverse", None, "list.reverse", False),
        ("unique", None, "list.unique", False),
        ("slice", None, "list.slice", False),
        ("head", None, "list.head", False),
        ("tail", None, "list.tail", False),
        ("sum", None, "list.sum", False),
        ("min", None, "list.min", False),
        ("max", None, "list.max", False),
        ("mean", None, "list.mean", False),
        ("median", None, "list.median", False),
        ("var", None, "list.var", False),
        ("std", None, "list.std", False),
        ("eval", None, "list.eval", False),
        ("drop_nulls", None, "list.drop_nulls", False),
        ("gather", None, "list.gather", False),
    ],
    "struct": [
        ("field", None, "struct.field", False),
        ("rename_fields", None, "struct.rename_fields", False),
        ("with_fields", None, "struct.with_fields", False),
        ("unnest", None, "struct.unnest", False),
        ("json_encode", None, "struct.json_encode", False),
        ("fields", None, "struct.fields", True),
    ],
}


def main() -> None:
    entries: dict[tuple[str, str], dict[str, object]] = {}
    for namespace, methods in METHODS.items():
        for name, pandas_path, polars_path, is_property in methods:
            key = (namespace, name)
            if key in entries:
                raise ValueError(f"Duplicate accessor method: {namespace}.{name}")
            entries[key] = {
                "pandas": pandas_path,
                "polars": polars_path,
                "property": is_property,
            }

    header = '''"""Generated by generate_methods.py; do not edit by hand."""

from __future__ import annotations

from typing import Optional, TypedDict


class AccessorMethodSpec(TypedDict):
    pandas: Optional[str]
    polars: Optional[str]
    property: bool


'''
    body = (
        "ACCESSOR_METHODS: dict[tuple[str, str], AccessorMethodSpec] = "
        + pprint.pformat(entries, sort_dicts=True, width=88)
        + "\n"
    )
    OUTPUT.write_text(header + body, encoding="utf-8")
    print(f"Wrote {len(entries)} accessor definitions to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
