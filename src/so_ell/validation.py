"""Data contracts for every table the pipeline hands from one step to the next.

Each stage validates its output before writing it, so a bad export or a code
change that breaks an assumption fails loudly at the stage that caused it
instead of showing up as a strange number in a chart. The event log is too
big to load whole for pandera, so it gets SQL checks; the smaller tables get
pandera schemas.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from so_ell import config

STUDENT_ID = r"^(F|W)\d{2}-(ALL|MW|TTh)-\d{4}$"

exam_attempts_schema = DataFrameSchema(
    {
        "semester_raw": Column(str, Check.isin(config.ANALYSIS_SEMESTERS)),
        "anon_student_id": Column(str, Check.str_matches(STUDENT_ID)),
        "session_id": Column(str),
        "midterm": Column(str, Check.isin(["mid1", "mid2"])),
        "attempt_start": Column("datetime64[us]"),
        "attempt_end": Column("datetime64[us]"),
        "attempt_minutes": Column(float, Check.ge(0)),
    },
    checks=Check(lambda df: (df["attempt_end"] >= df["attempt_start"]).all(), error="end before start"),
    unique=["semester_raw", "anon_student_id", "session_id", "midterm"],
)

student_scores_schema = DataFrameSchema(
    {
        "semester_raw": Column(str, Check.isin(config.ANALYSIS_SEMESTERS)),
        "anon_student_id": Column(str, Check.str_matches(STUDENT_ID)),
        "midterm": Column(str, Check.isin(["mid1", "mid2"])),
        "score_pct": Column(float, Check.in_range(0, 100)),
        "coverage_pct": Column(float, Check.in_range(0, 100)),
        "n_items_scored": Column(int, Check.ge(1)),
    },
    unique=config.WINDOW_KEYS,
)

features_schema = DataFrameSchema(
    {
        "score_pct": Column(float, Check.in_range(0, 100)),
        "parsons__mean_first_score": Column(float, Check.in_range(0, 1), nullable=True),
        "mchoice__mean_first_score": Column(float, Check.in_range(0, 1), nullable=True, required=False),
        "activecode__mean_best_score": Column(float, Check.in_range(0, 1), nullable=True, required=False),
        "all_activity__total_events": Column(float, Check.ge(0), coerce=True),
        "timing__share_last_7d": Column(float, Check.in_range(0, 1), nullable=True, required=False),
        "coverage__chapter_diversity": Column(float, Check.in_range(0, 1), nullable=True, required=False),
    },
    unique=config.WINDOW_KEYS,
)


def _coerce_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["semester_raw", "anon_student_id", "session_id", "midterm"]:
        if col in out:
            out[col] = out[col].astype(str)
    return out


def validate(df: pd.DataFrame, schema: DataFrameSchema) -> pd.DataFrame:
    schema.validate(_coerce_str(df), lazy=True)
    return df


def event_log_checks(events_path: Path = config.EVENTS_PARQUET) -> pd.DataFrame:
    """Row-level checks on the full event log, run in DuckDB. Returns a pass/fail table."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    src = f"read_parquet('{events_path.as_posix()}')"
    checks = {
        "no_null_student_id": f"SELECT COUNT(*) FROM {src} WHERE anon_student_id IS NULL",
        "student_id_format": f"SELECT COUNT(*) FROM {src} WHERE NOT regexp_matches(anon_student_id, '{STUDENT_ID}')",
        "no_null_selection": f"SELECT COUNT(*) FROM {src} WHERE selection IS NULL",
        "time_parsed": f"SELECT COUNT(*) FROM {src} WHERE time IS NULL",
        "time_in_plausible_range": f"SELECT COUNT(*) FROM {src} WHERE time < TIMESTAMP '2019-01-01' OR time > TIMESTAMP '2026-12-31'",
        "exact_duplicate_rows": f"""SELECT COUNT(*) - COUNT(DISTINCT (time, anon_student_id, action, problem_name, session_id, selection, input))
                                    FROM {src}""",
        "event_outside_semester_window": f"""
            WITH s AS (
                SELECT semester, quantile_cont(time, 0.001) lo, quantile_cont(time, 0.999) hi
                FROM {src} GROUP BY semester
            )
            SELECT COUNT(*) FROM {src} e JOIN s USING (semester)
            WHERE e.time < s.lo - INTERVAL 60 DAY OR e.time > s.hi + INTERVAL 60 DAY""",
    }
    # Some rows legitimately fail soft checks (unparseable timestamps exist in the
    # raw export). Hard checks must be zero; soft checks are reported, not fatal.
    hard = {"no_null_student_id", "student_id_format", "no_null_selection"}
    total = con.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
    rows = []
    for name, sql in checks.items():
        failures = con.execute(sql).fetchone()[0]
        rows.append(
            {
                "check": name,
                "severity": "hard" if name in hard else "soft",
                "failing_rows": int(failures),
                "failing_share": failures / total,
                "passed": failures == 0 or name not in hard,
            }
        )
    con.close()
    report = pd.DataFrame(rows)
    failed = report.loc[~report["passed"], "check"].tolist()
    if failed:
        raise ValueError(f"Event log failed hard checks: {failed}")
    return report
