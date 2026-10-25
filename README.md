# dfmacro

Native Python syntax for DataFrame expressions.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The problem

DataFrame expression APIs are verbose:

```python
# pandas
df[df["price"] > 100]

# polars
df.filter(pl.col("price") > 100)
```

You repeat `df`, or you memorize `pl.col`. Neither reads like the query you have in your head.

## The solution

```python
import dfmacro

@dfmacro.enable(df)
def result():
    return df[price > 100]
```

Write bare column names. Use native Python operators. No `pl.col`, no repeated `df`.

### Advanced querying

```python
@dfmacro.enable(df)
def analysis():
    return (
        df[price > 100]
        .mutate(
            double_price=price * 2,
            level=when(price > 150, "premium").otherwise("standard"),
        )
        .select(category, double_price, level)
        .groupby(category)
        .agg(avg_price=price.mean())
    )
```

### String and date operations

```python
@dfmacro.enable(df)
def time_series():
    return df[
        category.str.contains("elect") & (date.dt.year == 2024)
    ]
```

### SQL generation

```python
expr = price > 100
sql_where = dfmacro.compile_to_sql(expr)
# Output: '("price" > 100)'
```

The expression is lazy: it builds a tree, then compiles to the right backend when evaluated.

## Installation

```bash
pip install dfmacro
```

Requires Python 3.8+ and pandas 1.5+. Polars support is included.

## Supported features

- **Filtering**: `df[price > 100]`, `&`, `|`, `~`
- **Selection**: `df.select(price, category)`
- **Aggregation**: `df.groupby(category).agg(avg_price=price.mean())`
- **Mutation**: `df.mutate(double=price * 2)`
- **Conditional**: `when(price > 100, "high").otherwise("low")`
- **Joins**: `df.join(other, on=category, how="left")`
- **Sorting**: `df.order_by(price, descending=True)`
- **Row operations**: `head`, `tail`, `slice`, `drop`, `rename`
- **String accessors**: `.str.contains()`, `.str.lower()`, `.str.replace()`, `.str.extract()`, and more
- **Date accessors**: `.dt.year`, `.dt.month`, `.dt.dayofweek`, `.dt.strftime()`, and more
- **SQL generation**: `dfmacro.compile_to_sql(expr)`
- **Jupyter support**: `%load_ext dfmacro` + `%dfmacro df`

## How it works

Six layers, and you only ever see the friendliest one:

1. **Version dispatch** — Python 3.13+ uses the `f_locals` write-through proxy (PEP 667); 3.12 and earlier fall back to `PyFrame_LocalsToFast` via `ctypes`. All hidden inside the decorator.
2. **Frame interception** — `sys.settrace` injects at the function's `call` event and cleans up on `return` / `exception`.
3. **Name injection** — the target function's bytecode is scanned for `LOAD_GLOBAL` names. Column names are injected into `frame.f_globals`, because bare names compile to `LOAD_GLOBAL`, not `LOAD_FAST`.
4. **Expression tree** — `ColumnExpr` overloads Python operators to build a lazy tree. Nothing executes until compilation.
5. **Backend compilation** — `compile_expr` turns the tree into pandas or polars operations, with column validation and friendly errors.
6. **Error translation** — `and`/`or` misuse and operator precedence traps are converted into actionable messages, not silent bugs.

## Why not use `inspect.getsource` or a `with` block?

- **`inspect.getsource`** fails in interactive shells and breaks on nested scopes.
- **`with` blocks** require users to change indentation and pollute a named scope.
- **`dfmacro`** intercepts at the frame level, so the user's source code is never rewritten.

## Known limitations

- **Single-threaded assumption.** `sys.settrace` is process-global. Multi-threaded use needs additional guards; this is a future improvement.
- **`and` / `or` do not work.** Python forces truthiness evaluation, so use `&` / `|` and wrap comparisons in parentheses:
  ```python
  df[(price > 100) & (category == "electronics")]
  ```
- **Chained string selection.** `df[price > 100]["category"]` returns a native pandas Series, not a `DF`. Use `.select(category)` for chainable operations.

## Development

```bash
git clone https://github.com/Tony123-tech/dfmacro.git
cd dfmacro
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## License

MIT
