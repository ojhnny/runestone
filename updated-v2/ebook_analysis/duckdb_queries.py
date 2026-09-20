"""Thin DuckDB wrapper around the parquet log.

I didn't want to rewrite the 20M-row scoring in SQL (Parsons / ActiveCode
parsing is messy). DuckDB is for the stuff SQL is actually good at: joins,
window functions, and group-bys on the raw event table.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from ebook_analysis.paths import ATTEMPTS_PARQUET, EVENTS_PARQUET, SQL_DIR


def _sql_path(path: Path) -> str:
    # DuckDB wants forward slashes; quotes in the path would break the string.
    return path.resolve().as_posix().replace("'", "''")


def connect(
    events_path: Path = EVENTS_PARQUET,
    attempts_path: Path = ATTEMPTS_PARQUET,
) -> duckdb.DuckDBPyConnection:
    if not events_path.exists():
        raise FileNotFoundError(
            f"Missing {events_path}. Put runestone_event_log.parquet in the project root."
        )
    if not attempts_path.exists():
        raise FileNotFoundError(
            f"Missing {attempts_path}. Need exam_windows/timed_exam_attempts.parquet."
        )

    con = duckdb.connect()
    # 20M rows is fine on a laptop, but no reason to use a single thread.
    con.execute("PRAGMA threads=4")
    con.execute(
        f"CREATE OR REPLACE VIEW raw_events AS SELECT * FROM read_parquet('{_sql_path(events_path)}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW exam_attempts AS SELECT * FROM read_parquet('{_sql_path(attempts_path)}')"
    )
    return con


def load_sql(name: str) -> str:
    path = SQL_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def query(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute(load_sql(name)).df()


def run_pre_exam_views(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build the pre_exam_events view, then return a one-row count."""
    return query(con, "04_pre_exam_events.sql")


def build_sql_features(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, pd.DataFrame]:
    """Run the SQL feature queries and return named frames."""
    close_when_done = False
    if con is None:
        con = connect()
        close_when_done = True
    try:
        overview = query(con, "01_event_overview.sql")
        by_semester = query(con, "02_events_by_semester.sql")
        windows = query(con, "03_exam_windows.sql")
        pre_count = run_pre_exam_views(con)
        student_features = query(con, "05_student_exam_features.sql")
        diversity = query(con, "06_chapter_diversity.sql")
        chapter_cohort = query(con, "07_chapter_by_cohort.sql")
        timing_bins = query(con, "08_timing_bins.sql")
        family_volume = query(con, "09_family_volume.sql")

        student_features = student_features.merge(
            diversity,
            on=["semester_raw", "anon_student_id", "midterm"],
            how="left",
        )
        return {
            "overview": overview,
            "by_semester": by_semester,
            "windows": windows,
            "pre_count": pre_count,
            "student_features": student_features,
            "chapter_cohort": chapter_cohort,
            "timing_bins": timing_bins,
            "family_volume": family_volume,
        }
    finally:
        if close_when_done:
            con.close()
