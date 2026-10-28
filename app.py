"""Interactive, safe Gradio demo for dfmacro."""

from __future__ import annotations

import ast
import time
from typing import Any

import numpy as np
import pandas as pd

from dfmacro import DF, enable
from dfmacro.expr import BinaryExpr, ColumnExpr, Expr, Literal, UnaryExpr


def _make_demo_data() -> pd.DataFrame:
    """Create reproducible e-commerce data for interactive queries."""
    rng = np.random.default_rng(42)
    row_count = 10_000
    return pd.DataFrame(
        {
            "order_id": np.arange(1, row_count + 1),
            "price": np.round(rng.lognormal(mean=4.2, sigma=0.8, size=row_count), 2),
            "category": rng.choice(
                ["books", "electronics", "home", "clothing"], size=row_count
            ),
            "date": pd.Timestamp("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365, size=row_count), unit="D"),
        }
    )


_DATA = _make_demo_data()


def _parse_value(node: ast.AST, columns: set[str]) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in (
        bool,
        int,
        float,
        str,
        type(None),
    ):
        return Literal(node.value)
    if isinstance(node, ast.Name) and node.id in columns:
        return ColumnExpr(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _parse_value(node.operand, columns)
        if not isinstance(value, Literal) or type(value.value) not in (int, float):
            raise ValueError("Unary +/- is only allowed on numeric literals.")
        return Literal(-value.value if isinstance(node.op, ast.USub) else value.value)
    raise ValueError("Only column names and scalar literals are allowed in queries.")


def _parse_expression(node: ast.AST, columns: set[str]) -> Expr:
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = _parse_value(node.left, columns)
        right = _parse_value(node.comparators[0], columns)
        operators = {
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }
        operation = operators.get(type(node.ops[0]))
        if operation is None:
            raise ValueError("Query comparisons must use ==, !=, <, <=, >, or >=.")
        return BinaryExpr(operation, left, right)

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        left = _parse_expression(node.left, columns)
        right = _parse_expression(node.right, columns)
        return BinaryExpr("&" if isinstance(node.op, ast.BitAnd) else "|", left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return UnaryExpr("~", _parse_expression(node.operand, columns))

    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [_parse_expression(value, columns) for value in node.values]
        operation = "&" if isinstance(node.op, ast.And) else "|"
        result = values[0]
        for value in values[1:]:
            result = BinaryExpr(operation, result, value)
        return result

    raise ValueError(
        "Query must contain a supported comparison, optionally combined with "
        "&, |, and parentheses."
    )


def parse_query(source: str, columns: list[str]) -> Expr:
    """Parse only the supported `df[expression]` syntax; never execute Python."""
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Query syntax error: {exc.msg}.") from exc
    if (
        len(module.body) != 1
        or not isinstance(module.body[0], ast.Expr)
        or not isinstance(module.body[0].value, ast.Subscript)
    ):
        raise ValueError("Enter one filter expression, for example df[price > 100].")

    subscript = module.body[0].value
    if not isinstance(subscript.value, ast.Name) or subscript.value.id != "df":
        raise ValueError("The query must filter the demo DataFrame using df[...].")
    return _parse_expression(subscript.slice, set(columns))


def _to_backend_frame(backend: str) -> Any:
    if backend == "Pandas":
        return _DATA.copy()
    if backend == "Polars":
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError("Install the Polars extra to use this backend.") from exc
        return pl.from_pandas(_DATA)
    if backend == "DuckDB":
        return _DATA.copy()
    raise ValueError(f"Unknown backend {backend!r}.")


def run_query(source: str, backend: str) -> tuple[pd.DataFrame, str]:
    """Validate a filter, run it on the selected backend, and report duration."""
    start = time.perf_counter()
    try:
        expression = parse_query(source, list(_DATA.columns))
        frame = _to_backend_frame(backend)

        @enable(frame)
        def execute(df: Any, predicate: Expr, use_duckdb: bool) -> Any:
            if use_duckdb:
                return df.to_duckdb()[predicate].collect()
            return df[predicate]

        result = execute(frame, expression, backend == "DuckDB")
        raw_result = result.raw if isinstance(result, DF) else result
        if type(raw_result).__module__.startswith("polars"):
            raw_result = raw_result.to_pandas()
        elapsed = time.perf_counter() - start
        return raw_result, f"Completed in {elapsed:.4f} seconds ({backend})."
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return pd.DataFrame(), f"Query failed after {elapsed:.4f} seconds: {exc}"


def create_app() -> Any:
    """Build and return the Gradio Blocks application."""
    import gradio as gr

    with gr.Blocks(title="dfmacro DataFrame Query Demo") as demo:
        gr.Markdown(
            "# dfmacro Query Demo\n"
            "Filter 10,000 e-commerce rows using a safe subset of dfmacro syntax. "
            "Queries are parsed as expressions and arbitrary Python is not run."
        )
        query = gr.Textbox(
            label="Filter query",
            value="df[price > 100]",
            lines=2,
            placeholder="df[price > 100]",
        )
        backend = gr.Dropdown(
            choices=["Pandas", "Polars", "DuckDB"],
            value="Pandas",
            label="Backend",
        )
        run = gr.Button("Run", variant="primary")
        timing = gr.Textbox(label="Execution time", interactive=False)
        output = gr.Dataframe(label="Filtered rows", interactive=False)
        run.click(run_query, inputs=[query, backend], outputs=[output, timing])
    return demo


if __name__ == "__main__":
    create_app().launch()
