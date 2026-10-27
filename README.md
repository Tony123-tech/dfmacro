# dfmacro

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Write DataFrame queries with bare column names and ordinary Python syntax.**

`dfmacro` turns column references into lazy expressions, then compiles those
expressions for supported DataFrame backends.

## The problem

Pandas and Polars make column references explicit:

```python
# pandas
df[df["price"] > 100]

# polars
df.filter(pl.col("price") > 100)
```

For multi-step analysis, repeating those references can obscure the intent of a
query.

## The solution

Use `@enable` to rewrite a function's source so bare column names become
expressions:

```python
import pandas as pd
from dfmacro import enable

df = pd.DataFrame(
    {
        "price": [50, 150, 200, 80],
        "category": ["books", "electronics", "electronics", "books"],
    }
)

@enable(df)
def expensive_items():
    return df[price > 100]

result = expensive_items()
print(result.raw)
```

The decorated function returns a `DF` wrapper for DataFrame results. Use
`.raw` to access the underlying backend DataFrame.

## Advanced querying

Expressions can be combined with filtering, mutation, selection, and
aggregation:

```python
from dfmacro import enable, when

@enable(df)
def category_summary():
    return (
        df[price > 50]
        .mutate(
            double_price=price * 2,
            tier=when(price > 150, "premium").otherwise("standard"),
        )
        .select(category, double_price, tier)
        .groupby(category)
        .agg(avg_price=double_price.mean())
    )
```

Use `&`, `|`, and `~` to combine expressions. Parenthesize comparisons:

```python
@enable(df)
def selected_items():
    return df[(price > 100) & (category == "electronics")]
```

### String and date expressions

```python
@enable(df)
def matching_items():
    return df[
        category.str.contains("elect")
        & (created_at.dt.year == 2025)
    ]
```

### SQL generation

Build an expression directly and compile supported expressions to SQL:

```python
from dfmacro import ColumnExpr, compile_to_sql

predicate = ColumnExpr("price") > 100
sql_where = compile_to_sql(predicate)
print(sql_where)  # ("price" > 100)
```

`compile_to_sql` generates a SQL fragment; it does not execute the query.

## Extreme Performance: C & Rust Engines

`dfmacro` is built for scale. Beyond standard Pandas and Polars backends, it features optional extensions that bypass Python's GIL and utilize multi-core parallelism.

- **Rust Engine (`dfmacro_rs`)**: Powered by `Rayon` and `PyO3`, the Rust engine handles heavy numeric operations, multi-core filtering, string regex matching, and aggregations at native speed. It releases the GIL using `py.allow_threads()` to prevent bottlenecking.
- **C Engine (`dfmacro_c`)**: A true **Zero-Copy** extension. Instead of allocating new boolean masks, it directly scans memory pointers to return filtered indices instantly.

### Benchmark (50M Rows)

| Operation | Pandas (NumPy) | dfmacro (Rust / Rayon) | Speedup |
| :--- | :--- | :--- | :--- |
| **Sum / Mean** | ~0.35s (Single-core) | **~0.04s** (Multi-core) | **~8.7x faster** |
| **Filter (`> 500`)** | ~0.25s | **~0.22s** | **~1.2x faster** |

## Installation

The package declares backend integrations as optional extras. Install the
backend(s) you plan to use:

```bash
pip install 'dfmacro[pandas]'
pip install 'dfmacro[polars]'
pip install 'dfmacro[dask]'
pip install 'dfmacro[duckdb]'
```

Install multiple extras together when needed:

```bash
pip install 'dfmacro[pandas,polars,dask,duckdb]'
```

The `pandas` extra installs Pandas; the `polars`, `dask`, and `duckdb` extras
install their corresponding optional integrations.

## Supported features

### DataFrame verbs

The `DF` wrapper provides the following operations. Availability and behavior
can vary by backend; see [Backend support](#backend-support).

- **Filtering and selection:** `df[expression]`, `select`, `filter`, `where`,
  `mask`, `between`, `isin`
- **Mutation:** `mutate`, `assign`
- **Grouping and aggregation:** `groupby(...).agg(...)`
- **Sorting and row selection:** `order_by`, `sort_values`, `nlargest`,
  `nsmallest`, `head`, `tail`, `slice`, `sample`
- **Index and column operations:** `set_index`, `reset_index`, `drop`, `rename`
- **Null handling:** `drop_null`, `fill_null`, `ffill`, `bfill`, `interpolate`
- **Statistics:** `describe`, `value_counts`, `corr`, `cov`, `nunique`, `mode`,
  `skew`, `kurt`
- **Cumulative and shift operations:** `pct_change`, `shift`, `diff`, `cumsum`,
  `cummax`, `cummin`
- **Reshaping and time series:** `melt`, `pivot`, `pivot_table`, `rolling`,
  `ewm`, `resample`, `asfreq`
- **Joins and concatenation:** `join`, `merge`, `concat`, `merge_asof`,
  `cross_join`, `semi_join`, `anti_join`
- **Math and data helpers:** `apply`, `pipe`, `clip`, `abs`, `round`, `log`,
  `sqrt`, `pow`
- **Other utilities:** `explain`, `lazy`, `to_duckdb`

### Expressions and accessors

- **Operators:** `+`, `-`, `*`, `/`, comparisons, `&`, `|`, and `~`
- **Aggregations:** `mean`, `sum`, `min`, `max`, `count`, `median`, `std`,
  `var`, `first`, `last`, `nunique`, `quantile`
- **Window expressions:** `rank`, `cumsum`, `cummax`, `cummin`, `shift`, `diff`,
  `pct_change`
- **Math expressions:** `abs`, `round`, `sqrt`, `pow`
- **Conditional expressions:** `when(condition, value).otherwise(value)`
- **String accessor:** `.str.contains()`, `.startswith()`, `.endswith()`,
  `.lower()`, `.upper()`, `.strip()`, `.len()`, `.replace()`, `.split()`,
  `.slice()`, `.pad()`, `.zfill()`, `.match()`, `.extract()`, `.lstrip()`,
  `.rstrip()`, `.title()`, `.count()`, `.reverse()`, `.fullmatch()`,
  `.isalnum()`, `.isalpha()`, `.isdigit()`, `.islower()`, `.isupper()`
- **Datetime accessor:** `.dt.year`, `.month`, `.day`, `.hour`, `.minute`,
  `.second`, `.microsecond`, `.nanosecond`, `.dayofweek`, `.quarter`,
  `.dayofyear`, `.week`, `.days_in_month`, `.is_month_start`,
  `.is_month_end`, `.is_year_start`, `.is_year_end`, `.is_leap_year`,
  `.month_name`, `.day_name`, `.normalize()`, `.floor()`, `.ceil()`, `.round()`,
  `.strftime()`
- **Registry-backed accessors:** generated accessor metadata also describes
  categorical (`.cat`), numeric (`.num`), list (`.list`), and struct
  (`.struct`) operations. A registry entry is not a guarantee of support by
  every installed backend version; unsupported methods report a capability
  error.

## Backend support

| Backend | Status |
| --- | --- |
| Pandas | Core DataFrame wrapper and expression compiler. Install with `dfmacro[pandas]`. |
| Polars | Core DataFrame wrapper and expression compiler. Install with `dfmacro[polars]`. |
| Dask DataFrame | Lazy expression filtering, selection, mutation, and Dask-native sorting. Install with `dfmacro[dask]`. |
| DuckDB | SQL generation is available through `compile_to_sql`; the optional DuckDB wrapper is available with `dfmacro[duckdb]`. |

Dask expression evaluation reuses the Pandas compiler on each partition.
Operations that require cross-partition coordination, such as groupby
aggregation and joins, are not currently supported through the Dask `DF`
wrapper.

Some operations have backend-specific limitations:

- `apply` and `asfreq` are Pandas-only.
- Polars DataFrames do not have pandas-style indexes; `set_index` raises
  `NotImplementedError`, while `reset_index` returns a cloned DataFrame.
- Polars `resample` infers the time column only when exactly one `Date` or
  `Datetime` column exists; otherwise pass `on=`.
- Accessor methods are dispatched using configured backend paths. Methods
  without a path for the selected backend raise `CompileError`.

## Jupyter

The decorator needs retrievable Python source. In notebooks and other
interactive environments, load the extension and use the cell magic:

```python
%load_ext dfmacro
```

Then run a cell such as:

```python
%%dfmacro df
result = df[price > 100]
```

The magic rewrites the cell's source directly and does not require
`inspect.getsource` for the cell body.

## How it works

1. **AST rewriting** — `@enable` reads and rewrites the decorated function's
   source, converting bare column names into lookups on a private expression
   namespace.
2. **Scope resolution** — function arguments, locals, globals, builtins, and
   closure variables are accounted for so they are not accidentally treated as
   columns.
3. **Interactive execution** — `%dfmacro` rewrites cell source directly when
   decorator source retrieval is unsuitable. The library does not use
   `sys.settrace` to inject column names.
4. **Expression tree** — overloaded operators, accessors, and conditional
   helpers construct lazy expression nodes.
5. **Backend compilation** — the selected compiler translates expression nodes
   into Pandas or Polars operations; Dask expressions use Pandas compilation
   per partition.
6. **Error reporting** — missing columns and unsupported operations produce
   explicit errors rather than silently selecting a different backend path.

## Known limitations

- Python's `and` and `or` evaluate truthiness and cannot combine lazy
  expressions. Use `&` and `|`, and parenthesize each comparison.
- `@enable` requires inspectable source code. For Jupyter cells, use the
  `%dfmacro` magic. If source is unavailable, the decorator raises a
  `RuntimeError` with this guidance.
- Selecting a string column with `df["column"]` returns the native backend
  Series rather than a chainable `DF`. Use `df.select(column)` for DataFrame
  chaining.
- Dask currently supports only the documented partition-local operations and
  Dask-native sorting; not every `DF` method is available for Dask.
- SQL generation supports only the expression forms implemented by the SQL
  compiler; it is not a general-purpose query planner.

## Development

```bash
git clone https://github.com/Tony123-tech/dfmacro.git
cd dfmacro
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,pandas,polars]'
python -m pytest -q
```

## Build Extensions

```bash
# Build Rust extension
cd src_rust
maturin develop --release
cd ..

# Build C extension
cd src_c
python setup.py build_ext --inplace
cd ..
```
Regenerate the accessor capability registry after changing
[`generate_methods.py`](./generate_methods.py):

```bash
python generate_methods.py
```

## License

dfmacro is distributed under the MIT License. See [LICENSE](./LICENSE).
