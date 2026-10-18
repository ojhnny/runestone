"""Helpers for the build notebooks: show the exact code before running it.

The notebooks in ``notebooks/`` run the pipeline step by step. Each step
first prints the source of the function (or SQL file) it is about to call, so
a reader scrolling the notebook sees every line of code that produced the
results, while the code itself still lives in one tested package.
"""

from __future__ import annotations

import inspect

from IPython.display import Markdown, display

from so_ell import config


def show(*objects) -> None:
    """Render the source of functions, classes or constants as a code block."""
    for obj in objects:
        source = inspect.getsource(obj)
        where = f"{inspect.getmodule(obj).__name__}.{getattr(obj, '__qualname__', '')}".rstrip(".")
        display(Markdown(f"`{where}`\n```python\n{source.rstrip()}\n```"))


def show_sql(*names: str) -> None:
    """Render one or more files from ``sql/``."""
    for name in names:
        text = (config.SQL_DIR / name).read_text(encoding="utf-8")
        display(Markdown(f"`sql/{name}`\n```sql\n{text.rstrip()}\n```"))


def describe(df, columns=None):
    """Aggregate summary only (count/mean/quantiles), never row-level data."""
    cols = columns or df.select_dtypes("number").columns
    return df[cols].describe().T.round(3)
