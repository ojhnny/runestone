"""Two secondary analyses that need the raw log again.

1. Timed practice midterms. Instructors posted timed practice exams before
   each midterm ("Mid2-F25-Practice-Timed"). Did students who took one do
   better, and does a practice-exam score predict the real one?
2. Cramming vs. spacing. How much of a student's last-28-day practice fell
   in the final 3 / 7 days, compared between the top and bottom score quartiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from so_ell import config
from so_ell.config import WINDOW_KEYS
from so_ell.exam_windows import clean_name, parse_midterm
from so_ell.features import drop_exam_window_events
from so_ell.scores import add_event_scores
from so_ell.stats import within_group_quartile_gap, within_group_rank_corr

ATTEMPT_KEYS = ["semester_raw", "anon_student_id", "session_id", "midterm"]


def _read_events(columns: list[str], selections: list[str]) -> pd.DataFrame:
    events = pd.read_parquet(
        config.EVENTS_PARQUET,
        columns=columns,
        filters=[("selection", "in", selections), ("semester", "in", config.ANALYSIS_SEMESTERS)],
    ).rename(columns={"semester": "semester_raw"})
    return events.loc[events["time"].notna()]


def practice_midterm_analysis(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    actual = scores[WINDOW_KEYS + ["exam_start", "score_pct"]]
    markers = _read_events(
        ["time", "anon_student_id", "session_id", "problem_name", "selection", "action", "semester"], ["timedExam"]
    )
    markers["midterm"] = markers["problem_name"].map(parse_midterm)
    markers = markers.loc[markers["midterm"].notna() & markers["problem_name"].map(clean_name).str.contains("practice")]

    starts = markers.loc[markers["action"].eq("start")].groupby(ATTEMPT_KEYS)["time"].min().rename("practice_start")
    ends = markers.loc[markers["action"].eq("finish")].groupby(ATTEMPT_KEYS)["time"].max().rename("practice_end")
    names = markers.groupby(ATTEMPT_KEYS)["problem_name"].agg(lambda s: " | ".join(sorted(set(s)))).rename("practice_name")
    attempts = pd.concat([starts, ends, names], axis=1).reset_index().dropna(subset=["practice_start"])
    attempts = attempts.merge(actual[WINDOW_KEYS + ["exam_start"]], on=WINDOW_KEYS, how="inner")
    attempts = attempts.loc[attempts["practice_start"] < attempts["exam_start"]].copy()
    attempts["finished_before_exam"] = attempts["practice_end"].notna() & (attempts["practice_end"] < attempts["exam_start"])

    started = attempts.groupby(WINDOW_KEYS, as_index=False).agg(
        timed_practice_starts=("practice_name", "nunique"),
        timed_practice_finishes=("finished_before_exam", "sum"),
    )

    # Score finished practice attempts using only the self-grading item types.
    answers = _read_events(
        ["time", "anon_student_id", "session_id", "problem_name", "selection", "action", "input", "semester"],
        ["mChoice", "parsons", "hparsonsAnswer", "unittest"],
    )
    answers = add_event_scores(answers)
    finished = attempts.loc[attempts["finished_before_exam"]]
    labeled = answers.merge(
        finished[ATTEMPT_KEYS + ["practice_start", "practice_end", "practice_name"]],
        on=["semester_raw", "anon_student_id", "session_id"],
        how="inner",
    )
    labeled = labeled.loc[labeled["time"].between(labeled["practice_start"], labeled["practice_end"])]
    item_keys = ATTEMPT_KEYS + ["practice_name", "problem_name"]
    items = (
        labeled.sort_values(item_keys + ["time"])
        .groupby(item_keys, as_index=False)
        .agg(family=("family", "first"), last=("event_score", "last"), best=("event_score", "max"))
    )
    items["item_score"] = np.where(items["family"].eq("mchoice"), items["last"], items["best"])
    attempt_scores = items.groupby(ATTEMPT_KEYS + ["practice_name"], as_index=False).agg(
        practice_items=("problem_name", "nunique"), practice_score_pct=("item_score", "mean")
    )
    attempt_scores["practice_score_pct"] *= 100
    best = attempt_scores.groupby(WINDOW_KEYS, as_index=False).agg(best_practice_score_pct=("practice_score_pct", "max"))

    student = actual.merge(started, on=WINDOW_KEYS, how="left").merge(best, on=WINDOW_KEYS, how="left")
    student[["timed_practice_starts", "timed_practice_finishes"]] = (
        student[["timed_practice_starts", "timed_practice_finishes"]].fillna(0).astype(int)
    )
    student["practice_group"] = np.select(
        [student["timed_practice_starts"].eq(0), student["timed_practice_finishes"].gt(0)],
        ["No timed practice", "Finished timed practice"],
        default="Started only",
    )
    student["cohort"] = student["semester_raw"].astype(str) + "_" + student["midterm"].astype(str)

    group = student.groupby(["midterm", "practice_group"], as_index=False).agg(
        n_student_exams=("anon_student_id", "count"),
        mean_exam_score=("score_pct", "mean"),
        median_exam_score=("score_pct", "median"),
    )
    cohort = student.groupby(["semester_raw", "midterm"], as_index=False).agg(
        n_student_exams=("anon_student_id", "count"),
        timed_practice_rate=("timed_practice_starts", lambda s: 100 * s.gt(0).mean()),
        finished_practice_rate=("timed_practice_finishes", lambda s: 100 * s.gt(0).mean()),
        mean_exam_score=("score_pct", "mean"),
        mean_best_practice_score=("best_practice_score_pct", "mean"),
    )
    pivot = student.pivot_table(index="cohort", columns="practice_group", values="score_pct", aggfunc="mean")
    finish_gap = (
        (pivot["Finished timed practice"] - pivot["No timed practice"]).mean()
        if {"Finished timed practice", "No timed practice"} <= set(pivot.columns)
        else np.nan
    )
    signals = pd.DataFrame(
        [
            {"metric": "practice_score_rank_corr", "value": within_group_rank_corr(student, "cohort", "best_practice_score_pct", "score_pct"),
             "description": "Within-cohort rank correlation, best timed-practice score vs real midterm score."},
            {"metric": "practice_score_quartile_gap", "value": within_group_quartile_gap(student, "cohort", "best_practice_score_pct", "score_pct"),
             "description": "Top minus bottom quartile of practice score, mean real midterm points."},
            {"metric": "finished_vs_none_gap", "value": finish_gap,
             "description": "Mean within-cohort gap: finished a timed practice exam vs did none."},
            {"metric": "finished_practice_attempts", "value": float(len(finished)),
             "description": "Finished timed practice attempts before the real exam."},
        ]
    )
    return {"student": student, "attempts": attempt_scores, "group": group, "cohort": cohort, "signals": signals}


def cramming_analysis(scores: pd.DataFrame, windows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from so_ell.scoring import EXAM_SELECTIONS

    events = _read_events(["time", "anon_student_id", "selection", "semester"], EXAM_SELECTIONS)
    events = drop_exam_window_events(events, windows)
    events["event_day"] = events["time"].dt.floor("D")
    pre = events[["semester_raw", "anon_student_id", "event_day"]].merge(
        scores[WINDOW_KEYS + ["exam_start", "score_pct"]], on=["semester_raw", "anon_student_id"], how="inner"
    )
    pre["days_before"] = (pre["exam_start"].dt.floor("D") - pre["event_day"]).dt.days
    pre = pre.loc[pre["days_before"].between(1, 28)].copy()
    pre["cohort"] = pre["semester_raw"].astype(str) + "_" + pre["midterm"].astype(str)

    band = scores[WINDOW_KEYS + ["score_pct"]].copy()
    band["cohort"] = band["semester_raw"].astype(str) + "_" + band["midterm"].astype(str)
    band["score_group"] = band.groupby("cohort")["score_pct"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    )
    pre = pre.merge(band[WINDOW_KEYS + ["score_group"]], on=WINDOW_KEYS)

    daily = pre.groupby(WINDOW_KEYS + ["score_group", "days_before"], as_index=False, observed=True).size()
    daily["share"] = daily["size"] / daily.groupby(WINDOW_KEYS)["size"].transform("sum")
    metrics = daily.groupby(WINDOW_KEYS + ["score_group"], as_index=False, observed=True).apply(
        lambda g: pd.Series(
            {
                "active_days_28d": g["days_before"].nunique(),
                "last3_share": g.loc[g["days_before"] <= 3, "share"].sum(),
                "last7_share": g.loc[g["days_before"] <= 7, "share"].sum(),
                "concentration_hhi": (g["share"] ** 2).sum(),
            }
        ),
        include_groups=False,
    )
    daily["timing_bin"] = pd.cut(
        daily["days_before"], bins=[0, 3, 7, 14, 28],
        labels=["1-3 days before", "4-7 days before", "8-14 days before", "15-28 days before"],
    )
    # Per student: share of their 28-day practice in each bin (0 when empty), then average by band.
    per_student = (
        daily.groupby(WINDOW_KEYS + ["timing_bin"], observed=False, as_index=False)["share"].sum()
        .merge(band[WINDOW_KEYS + ["score_group"]], on=WINDOW_KEYS)
    )
    per_student = per_student.merge(metrics[WINDOW_KEYS], on=WINDOW_KEYS)  # only students with activity
    bins = (
        per_student.groupby(["score_group", "timing_bin"], observed=True, as_index=False)["share"].mean()
        .rename(columns={"share": "mean_share_pct"})
    )
    bins["mean_share_pct"] *= 100
    summary = metrics.groupby("score_group", observed=True, as_index=False).agg(
        n_student_exams=("anon_student_id", "count"),
        mean_active_days_28d=("active_days_28d", "mean"),
        mean_last3_share=("last3_share", "mean"),
        mean_last7_share=("last7_share", "mean"),
        mean_concentration_hhi=("concentration_hhi", "mean"),
    )
    return {"bins": bins, "metrics": metrics, "summary": summary}
