"""Pre-exam practice features (pandas pass).

For every (student, midterm) we summarize the practice they did in the ebook
*before* that exam started. Two leakage guards:

1. Events inside ANY of the student's timed exam windows are removed first,
   so midterm 1 answers never count as practice for midterm 2.
2. Only events strictly before the exam start are used. ``cutoff_days`` moves
   that line earlier (e.g. 14 = only practice at least two weeks out), which
   is what the early-warning analysis uses.

Columns are named ``family__metric`` (``parsons__mean_first_score``), where
family is an activity type and ``all_activity`` / ``learning`` are roll-ups.
"""

from __future__ import annotations

import pandas as pd

from so_ell.config import WINDOW_KEYS
from so_ell.scores import add_event_scores, infer_answer_keys
from so_ell.scoring import KEYED_FAMILIES, SCOREABLE_FAMILIES


def drop_exam_window_events(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Remove every event that falls inside any of that student's exam windows."""
    events = events.reset_index(drop=True)
    probe = events[["semester_raw", "anon_student_id", "time"]].reset_index().merge(
        windows[["semester_raw", "anon_student_id", "exam_start", "exam_end"]],
        on=["semester_raw", "anon_student_id"],
        how="inner",
    )
    inside = probe.loc[probe["time"].between(probe["exam_start"], probe["exam_end"], inclusive="both"), "index"]
    return events.loc[~events.index.isin(inside.unique())].copy()


def score_study_events(raw: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Practice events (exam windows removed) with per-event scores attached."""
    study = add_event_scores(raw)
    study = drop_exam_window_events(study, windows)
    keys = infer_answer_keys(study)
    study = study.merge(keys[["problem_name", "answer_key"]], on="problem_name", how="left")
    keyed = study["family"].isin(KEYED_FAMILIES) & study["response_norm"].notna()
    study.loc[keyed, "event_score"] = (study.loc[keyed, "response_norm"] == study.loc[keyed, "answer_key"]).astype(float)
    return study


def summarize_problems(problem_pre: pd.DataFrame) -> pd.DataFrame:
    """First / last / best score per (student, exam, family, problem).

    groupby.first/last skip NaN, so "first" is the first *scored* attempt.
    Opening a Parsons / ActiveCode problem and never getting a score counts as 0.
    """
    keys = WINDOW_KEYS + ["family", "problem_name"]
    ordered = problem_pre.sort_values(keys + ["time"])
    g = ordered.groupby(keys, observed=True)["event_score"]
    out = pd.DataFrame(
        {
            "problem_events": g.size(),
            "first_score": g.first(),
            "last_score": g.last(),
            "best_score": g.max(),
        }
    ).reset_index()
    unscored = out["best_score"].isna() & out["family"].isin(["parsons", "activecode"])
    out.loc[unscored, ["first_score", "last_score", "best_score"]] = 0.0
    out["score_gain"] = out["last_score"] - out["first_score"]
    return out


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index=WINDOW_KEYS, columns="family")
    wide.columns = [f"{family}__{metric}" for metric, family in wide.columns]
    return wide.reset_index()


def build_practice_features(
    study: pd.DataFrame,
    windows: pd.DataFrame,
    student_scores: pd.DataFrame,
    cutoff_days: int = 0,
) -> pd.DataFrame:
    pre = study[["semester_raw", "anon_student_id", "time", "problem_name", "family", "event_score"]].merge(
        windows[WINDOW_KEYS + ["exam_start"]], on=["semester_raw", "anon_student_id"], how="inner"
    )
    cutoff = pre["exam_start"] - pd.to_timedelta(cutoff_days, unit="D")
    pre = pre.loc[pre["time"] < cutoff].copy()
    pre["event_day"] = pre["time"].dt.floor("D")

    def activity(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        return frame.groupby(by, as_index=False, observed=True).agg(
            total_events=("problem_name", "size"),
            active_days=("event_day", "nunique"),
            unique_problems=("problem_name", "nunique"),
            scored_events=("event_score", lambda s: s.notna().sum()),
            avg_event_score=("event_score", "mean"),
        )

    # conceptcheck (selectquestion) never carries a score, so its avg is all-NaN by design.
    row_activity = pd.concat(
        [activity(pre, WINDOW_KEYS + ["family"]), activity(pre, WINDOW_KEYS).assign(family="all_activity")],
        ignore_index=True,
    )

    per_problem = summarize_problems(pre.loc[pre["family"].isin(SCOREABLE_FAMILIES)])

    def quality(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        return frame.groupby(by, as_index=False, observed=True).agg(
            problems_seen=("problem_name", "nunique"),
            scored_problems=("best_score", lambda s: s.notna().sum()),
            mean_first_score=("first_score", "mean"),
            mean_last_score=("last_score", "mean"),
            mean_best_score=("best_score", "mean"),
            mean_score_gain=("score_gain", "mean"),
            mean_events_per_problem=("problem_events", "mean"),
        )

    problem_quality = pd.concat(
        [quality(per_problem, WINDOW_KEYS + ["family"]), quality(per_problem, WINDOW_KEYS).assign(family="learning")],
        ignore_index=True,
    )

    features = student_scores.merge(_wide(row_activity), on=WINDOW_KEYS, how="left").merge(
        _wide(problem_quality), on=WINDOW_KEYS, how="left"
    )
    # A student with no practice of a type did zero events of it (not "unknown").
    count_cols = [c for c in features.columns if c.endswith(("__total_events", "__active_days", "__unique_problems", "__scored_events", "__problems_seen", "__scored_problems"))]
    features[count_cols] = features[count_cols].fillna(0)
    features["cohort"] = features["semester_raw"].astype(str) + "_" + features["midterm"].astype(str)
    features["cutoff_days"] = cutoff_days
    return features


def mid1_to_mid2_changes(features: pd.DataFrame) -> pd.DataFrame:
    """One row per student who took both midterms, with mid1, mid2 and change columns."""
    numeric = [c for c in features.columns if "__" in c and pd.api.types.is_numeric_dtype(features[c])]
    base = features[["semester_raw", "anon_student_id", "midterm", "score_pct"] + numeric]
    wide = base.set_index(["semester_raw", "anon_student_id", "midterm"]).unstack("midterm")
    wide.columns = [f"{metric}__{midterm}" for metric, midterm in wide.columns]
    wide = wide.reset_index()
    wide = wide.loc[wide["score_pct__mid1"].notna() & wide["score_pct__mid2"].notna()].copy()
    wide["score_change"] = wide["score_pct__mid2"] - wide["score_pct__mid1"]
    changes = {
        f"{f}__change": wide[f"{f}__mid2"] - wide[f"{f}__mid1"]
        for f in numeric
        if f"{f}__mid1" in wide and f"{f}__mid2" in wide
    }
    return pd.concat([wide, pd.DataFrame(changes, index=wide.index)], axis=1)
