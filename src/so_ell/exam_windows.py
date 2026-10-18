"""Find each student's real, timed midterm attempt in the event log.

Runestone logs a `timedExam` event with action `start` / `finish` when a
student opens and submits a timed assessment. The problem name tells us which
assessment it was, but the naming is inconsistent across semesters
("Midterm1-Fall21-Timed", "Mid2-F25-Timed", "Mid1-PI-Timed", ...) and the same
mechanism is used for practice exams and in-class quizzes.

Rules, in order:
1. The name must mention midterm 1 or 2 ("m1", "mid 1", "midterm2", ...).
2. Names containing practice / post / quiz / study / intro / PI are not the
   real exam. "Mid2-F25-Practice-Timed" is excluded; "Mid2-F25-Timed-Accom"
   (accommodation version of the real exam) is kept.
3. One attempt = one (student, session, midterm). Start is the earliest start
   marker, end is the latest finish marker; a missing finish uses the start.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from so_ell import config

SEMESTER_LABELS = {"F": "Fall", "W": "Winter"}
MIDTERM_LABELS = {"mid1": "Midterm 1", "mid2": "Midterm 2"}
NOT_REAL_EXAM_PATTERN = r"\b(practice|post|quiz|study|intro|pi\w*)\b"


def clean_name(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def parse_midterm(name: object) -> str | None:
    cleaned = clean_name(name)
    if re.search(r"\b(?:m|mid|midterm)\s*1\b", cleaned):
        return "mid1"
    if re.search(r"\b(?:m|mid|midterm)\s*2\b", cleaned):
        return "mid2"
    return None


def is_real_exam_name(name: object) -> bool:
    cleaned = clean_name(name)
    if parse_midterm(cleaned) is None:
        return False
    return not re.search(NOT_REAL_EXAM_PATTERN, cleaned)


def semester_label(code: str) -> str | None:
    match = re.fullmatch(r"([FW])(\d{2})", str(code))
    if not match:
        return None
    term, yy = match.groups()
    return f"{SEMESTER_LABELS[term]} {2000 + int(yy)}"


def load_timed_markers(events_path: Path = config.EVENTS_PARQUET) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    markers = con.execute(
        f"""
        SELECT time, semester AS semester_raw, anon_student_id, session_id, problem_name, action
        FROM read_parquet('{events_path.as_posix()}')
        WHERE selection = 'timedExam' AND action IN ('start', 'finish') AND time IS NOT NULL
        """
    ).df()
    con.close()
    return markers


def build_attempts(markers: pd.DataFrame) -> pd.DataFrame:
    markers = markers.copy()
    markers["midterm"] = markers["problem_name"].map(parse_midterm)
    markers["is_real_exam"] = markers["problem_name"].map(is_real_exam_name)
    real = markers.loc[markers["midterm"].notna() & markers["is_real_exam"]]

    keys = ["semester_raw", "anon_student_id", "session_id", "midterm"]
    starts = real.loc[real["action"].eq("start")].groupby(keys)["time"].min().rename("attempt_start")
    ends = real.loc[real["action"].eq("finish")].groupby(keys)["time"].max().rename("attempt_end")
    names = real.groupby(keys)["problem_name"].agg(lambda s: " | ".join(sorted(set(s)))).rename("anchor_problem_name")

    attempts = pd.concat([starts, ends, names], axis=1).reset_index()
    attempts = attempts.loc[attempts["attempt_start"].notna()].copy()
    attempts["attempt_end"] = attempts["attempt_end"].fillna(attempts["attempt_start"])
    # A finish logged before the start belongs to an earlier session; ignore it.
    attempts["attempt_end"] = attempts[["attempt_start", "attempt_end"]].max(axis=1)
    attempts["semester"] = attempts["semester_raw"].map(semester_label)
    attempts["normalized_midterm_name"] = attempts["semester"] + " " + attempts["midterm"].map(MIDTERM_LABELS)
    attempts["attempt_minutes"] = (attempts["attempt_end"] - attempts["attempt_start"]).dt.total_seconds() / 60
    return attempts.sort_values(keys).reset_index(drop=True)


def anchor_name_audit(markers: pd.DataFrame) -> pd.DataFrame:
    """Every timed-exam name with a midterm number, and whether it was kept.

    This is the table to read when someone asks "how do you know which
    timed exam was the real midterm?"
    """
    use = markers.loc[markers["action"].eq("start")].copy()
    use["midterm"] = use["problem_name"].map(parse_midterm)
    use = use.loc[use["midterm"].notna()]
    use["kept_as_real_exam"] = use["problem_name"].map(is_real_exam_name)
    return (
        use.groupby(["semester_raw", "problem_name", "midterm", "kept_as_real_exam"], as_index=False)
        .agg(students=("anon_student_id", "nunique"), first_start=("time", "min"), last_start=("time", "max"))
        .sort_values(["semester_raw", "midterm", "students"], ascending=[True, True, False])
    )


def run_exam_windows(
    events_path: Path = config.EVENTS_PARQUET,
    out_path: Path = config.EXAM_ATTEMPTS_PARQUET,
) -> dict[str, pd.DataFrame]:
    markers = load_timed_markers(events_path)
    attempts = build_attempts(markers)
    attempts = attempts.loc[attempts["semester_raw"].isin(config.ANALYSIS_SEMESTERS)].copy()
    attempts.to_parquet(out_path, index=False)

    summary = (
        attempts.groupby(["semester_raw", "semester", "midterm"], as_index=False)
        .agg(
            attempts=("session_id", "size"),
            students=("anon_student_id", "nunique"),
            median_minutes=("attempt_minutes", "median"),
            first_start=("attempt_start", "min"),
            last_start=("attempt_start", "max"),
        )
    )
    expected = pd.MultiIndex.from_product(
        [[s for s in config.ANALYSIS_SEMESTERS if s in set(markers["semester_raw"])], ["mid1", "mid2"]],
        names=["semester_raw", "midterm"],
    ).to_frame(index=False)
    missing = expected.merge(summary[["semester_raw", "midterm"]], how="left", indicator=True)
    missing = missing.loc[missing["_merge"].eq("left_only")].drop(columns="_merge")
    return {"attempts": attempts, "summary": summary, "missing": missing, "name_audit": anchor_name_audit(markers)}
