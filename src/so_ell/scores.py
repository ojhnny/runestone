"""Reconstruct each student's midterm score from what they did inside the exam.

The log has no gradebook. What it has is every answer a student submitted
while their timed midterm was open. For each (student, exam, problem) we take:

- multiple choice: the last submitted answer's correct / incorrect flag
- Parsons (drag-the-lines-into-order): 1 if any check was correct, else 0
- ActiveCode (write code, run unit tests): best unit-test pass rate, else 0
- fill-in / clickable / drag-and-drop: these don't log correctness, so the
  answer key is inferred as the modal final answer across students, and only
  used when at least KEY_CONFIDENCE_THRESHOLD of students agree on it
- short answer: free text, never scored

score_pct is the mean over scored problems, so it is a reconstructed
percentage, not the official grade. coverage_pct says how many of the
problems a student touched could actually be scored.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from so_ell import config
from so_ell.config import WINDOW_KEYS
from so_ell.scoring import (
    EXAM_SELECTIONS,
    KEYED_FAMILIES,
    activity_family,
    normalize_response,
    parse_hparsons_score,
    parse_mchoice_score,
    parse_parsons_score,
    parse_unittest_pct,
)

# shortanswer is in here so it shows up as "observed but unsupported" in coverage.
SCORED_SELECTIONS = [s for s in EXAM_SELECTIONS if s != "selectquestion"]
KEYED_SELECTIONS = ["fillb", "clickableArea", "dragNdrop", "dragNdrop-drop"]
PARSED_SELECTIONS = ["mChoice", "parsons", "hparsonsAnswer", "unittest"] + KEYED_SELECTIONS


def load_scoreable_events(events_path: Path = config.EVENTS_PARQUET) -> pd.DataFrame:
    """Only the answer-type events; page views etc. never feed a score."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    selections = ", ".join(f"'{s}'" for s in EXAM_SELECTIONS)
    semesters = ", ".join(f"'{s}'" for s in config.ANALYSIS_SEMESTERS)
    parsed = ", ".join(f"'{s}'" for s in PARSED_SELECTIONS)
    keyed = ", ".join(f"'{s}'" for s in KEYED_SELECTIONS)
    raw = con.execute(
        f"""
        SELECT time, anon_student_id, problem_name, session_id, selection, semester AS semester_raw,
               -- action / input text is only parsed for these selections; dropping it
               -- elsewhere keeps 16M rows in a few GB instead of ~10.
               CASE WHEN selection IN ({parsed}) THEN action END AS action,
               CASE WHEN selection IN ({keyed}) THEN input END AS input
        FROM read_parquet('{events_path.as_posix()}')
        WHERE selection IN ({selections}) AND semester IN ({semesters}) AND time IS NOT NULL
        """
    ).df()
    con.close()
    for col in ["anon_student_id", "session_id", "selection", "semester_raw"]:
        raw[col] = raw[col].astype("category")
    return raw


def build_windows(attempts: pd.DataFrame) -> pd.DataFrame:
    """Collapse a student's attempts (e.g. a crashed and resumed session) into one window."""
    return (
        attempts.groupby(WINDOW_KEYS + ["semester", "normalized_midterm_name"], as_index=False, observed=True)
        .agg(
            exam_start=("attempt_start", "min"),
            exam_end=("attempt_end", "max"),
            n_sessions=("session_id", "nunique"),
            total_exam_minutes=("attempt_minutes", "sum"),
        )
        .sort_values(WINDOW_KEYS)
        .reset_index(drop=True)
    )


def add_event_scores(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["family"] = events["selection"].astype(str).map(activity_family)
    events["event_score"] = np.nan
    parsers = {
        "mChoice": parse_mchoice_score,
        "parsons": parse_parsons_score,
        "hparsonsAnswer": parse_hparsons_score,
        "unittest": parse_unittest_pct,
    }
    for selection, parser in parsers.items():
        mask = events["selection"].eq(selection)
        if mask.any():
            events.loc[mask, "event_score"] = events.loc[mask, "action"].map(parser).astype(float)
    keyed = events["family"].isin(KEYED_FAMILIES)
    source = events["input"].where(events["input"].notna(), events["action"])
    events["response_norm"] = pd.Series(None, index=events.index, dtype="object")
    if keyed.any():
        events.loc[keyed, "response_norm"] = source[keyed].map(normalize_response)
    return events


def infer_answer_keys(events: pd.DataFrame) -> pd.DataFrame:
    """Modal final answer per problem, with the share of students who gave it."""
    cols = ["problem_name", "answer_key", "key_n", "key_support", "key_confidence"]
    keyed = events.loc[events["family"].isin(KEYED_FAMILIES) & events["response_norm"].notna()]
    if keyed.empty:
        return pd.DataFrame(columns=cols)
    final = (
        keyed.sort_values(["problem_name", "anon_student_id", "time"])
        .groupby(["problem_name", "anon_student_id"], observed=True)
        .tail(1)
    )
    counts = final.groupby(["problem_name", "response_norm"], as_index=False).size().rename(columns={"size": "key_support"})
    totals = final.groupby("problem_name", as_index=False).size().rename(columns={"size": "key_n"})
    top = (
        counts.sort_values(["problem_name", "key_support", "response_norm"], ascending=[True, False, True])
        .groupby("problem_name")
        .head(1)
        .rename(columns={"response_norm": "answer_key"})
    )
    lookup = top.merge(totals, on="problem_name")
    lookup["key_confidence"] = lookup["key_support"] / lookup["key_n"]
    return lookup[cols]


def label_exam_events(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    labeled = events.merge(
        windows[WINDOW_KEYS + ["semester", "normalized_midterm_name", "exam_start", "exam_end"]],
        on=["semester_raw", "anon_student_id"],
        how="inner",
    )
    inside = labeled["time"].between(labeled["exam_start"], labeled["exam_end"], inclusive="both")
    return labeled.loc[inside].copy()


def _score_problem(group: pd.DataFrame, key_threshold: float) -> dict[str, object]:
    family = group["family"].mode().iloc[0]
    score, source, key_conf = np.nan, "unscored", np.nan
    values = group["event_score"].dropna()
    if family == "mchoice":
        if not values.empty:
            score, source = values.iloc[-1], "last_mchoice_result"
    elif family == "parsons":
        score, source = (values.max(), "best_parsons_result") if not values.empty else (0.0, "parsons_no_success")
    elif family == "activecode":
        tests = group.loc[group["selection"].eq("unittest"), "event_score"].dropna()
        score, source = (tests.max(), "best_unittest_percent") if not tests.empty else (0.0, "activecode_no_passing_test")
    elif family in KEYED_FAMILIES:
        finals = group.loc[group["response_norm"].notna()]
        if finals.empty:
            source = f"{family}_no_response"
        else:
            last = finals.iloc[-1]
            key_conf = last["key_confidence"]
            if pd.notna(last["answer_key"]) and pd.notna(key_conf) and key_conf >= key_threshold:
                score, source = float(last["response_norm"] == last["answer_key"]), f"{family}_modal_key"
            else:
                source = f"{family}_low_confidence_key"
    elif family == "shortanswer":
        source = "unsupported_shortanswer"
    return {
        "item_family": family,
        "score": score,
        "score_source": source,
        "key_confidence": key_conf,
        "n_events": len(group),
    }


def score_exams(
    raw: pd.DataFrame,
    windows: pd.DataFrame,
    key_threshold: float = config.KEY_CONFIDENCE_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    exam_events = add_event_scores(raw.loc[raw["selection"].isin(SCORED_SELECTIONS)])
    exam_events = label_exam_events(exam_events, windows)
    keys = infer_answer_keys(exam_events)
    exam_events = exam_events.merge(keys, on="problem_name", how="left")

    group_cols = WINDOW_KEYS + ["semester", "normalized_midterm_name", "problem_name"]
    exam_events = exam_events.sort_values(group_cols + ["time"])
    rows = []
    for key, group in exam_events.groupby(group_cols, observed=True, sort=False):
        rows.append(dict(zip(group_cols, key, strict=True)) | _score_problem(group, key_threshold))
    problem_scores = pd.DataFrame(rows)

    student_scores = (
        problem_scores.groupby(WINDOW_KEYS + ["semester", "normalized_midterm_name"], as_index=False, observed=True)
        .agg(
            n_items_observed=("problem_name", "nunique"),
            n_items_scored=("score", lambda s: s.notna().sum()),
            score_points=("score", "sum"),
            score_pct=("score", "mean"),
        )
        .merge(windows, on=WINDOW_KEYS + ["semester", "normalized_midterm_name"], how="left")
    )
    student_scores["score_pct"] *= 100
    student_scores["coverage_pct"] = 100 * student_scores["n_items_scored"] / student_scores["n_items_observed"]
    student_scores = student_scores.loc[student_scores["score_pct"].notna()].reset_index(drop=True)

    semester_summary = (
        student_scores.groupby(["semester_raw", "semester", "midterm"], as_index=False)
        .agg(
            n_students=("anon_student_id", "nunique"),
            mean_score_pct=("score_pct", "mean"),
            median_score_pct=("score_pct", "median"),
            sd_score_pct=("score_pct", "std"),
            mean_coverage_pct=("coverage_pct", "mean"),
        )
        .sort_values(["semester_raw", "midterm"])
    )
    unsupported = problem_scores.loc[
        problem_scores["score"].isna(),
        ["semester_raw", "midterm", "problem_name", "item_family", "score_source", "key_confidence", "n_events"],
    ]
    unsupported_summary = (
        unsupported.groupby(["item_family", "score_source"], as_index=False)
        .agg(student_problem_rows=("problem_name", "size"), distinct_problems=("problem_name", "nunique"))
        .sort_values("student_problem_rows", ascending=False)
    )
    return {
        "problem_scores": problem_scores,
        "student_scores": student_scores,
        "semester_summary": semester_summary,
        "answer_keys": keys.sort_values(["key_confidence", "key_n"], ascending=[True, False]),
        "unsupported_summary": unsupported_summary,
    }
