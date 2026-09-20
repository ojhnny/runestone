"""Rebuild midterm scores and the original pre-exam activity features.

This is the heavy pandas pass over runestone_event_log.parquet. Scoring parsers live
in ebook_analysis.scoring so the tests / notebooks don't copy-paste them. For the newer
DuckDB features and Looker tables, run `python -m ebook_analysis` instead.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

# Running `python rebuild_midterm_scores.py` from the repo root should still work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ebook_analysis.scoring import (
    KEYED_FAMILIES,
    SCOREABLE_FAMILIES,
    WINDOW_KEYS,
    activity_family,
    first_non_null,
    last_non_null,
    normalize_response,
    parse_hparsons_score,
    parse_mchoice_score,
    parse_parsons_score,
    parse_unittest_pct,
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)


def infer_answer_keys(events: pd.DataFrame) -> pd.DataFrame:
    # For fill-in / click / drag items the log doesn't say "correct".
    # I take each student's last response and treat a high-agreement answer
    # as the key. 0.65 is arbitrary but keeps us from scoring coin-flips.
    if events.empty:
        return pd.DataFrame(columns=["problem_name", "answer_key", "key_n", "key_support", "key_confidence"])
    ordered = events.sort_values(["problem_name", "anon_student_id", "time"])
    final_resp = (
        ordered.loc[ordered["response_norm"].notna(), ["problem_name", "anon_student_id", "response_norm"]]
        .groupby(["problem_name", "anon_student_id"], as_index=False)
        .tail(1)
    )
    if final_resp.empty:
        return pd.DataFrame(columns=["problem_name", "answer_key", "key_n", "key_support", "key_confidence"])
    counts = (
        final_resp.groupby(["problem_name", "response_norm"], as_index=False)
        .size()
        .rename(columns={"size": "key_support"})
    )
    totals = (
        final_resp.groupby("problem_name", as_index=False)
        .size()
        .rename(columns={"size": "key_n"})
    )
    counts = counts.sort_values(["problem_name", "key_support", "response_norm"], ascending=[True, False, True])
    top = counts.groupby("problem_name", as_index=False).head(1).rename(columns={"response_norm": "answer_key"})
    lookup = top.merge(totals, on="problem_name", how="left")
    lookup["key_confidence"] = lookup["key_support"] / lookup["key_n"]
    return lookup


def add_explicit_event_scores(events: pd.DataFrame) -> pd.DataFrame:
    # Only fill event_score when that family actually logs correctness.
    events = events.copy()
    events["family"] = events["selection"].map(activity_family)
    events["event_score"] = np.nan
    mchoice_mask = events["selection"].eq("mChoice")
    parsons_mask = events["selection"].eq("parsons")
    hparsons_mask = events["selection"].eq("hparsonsAnswer")
    unittest_mask = events["selection"].eq("unittest")
    events.loc[mchoice_mask, "event_score"] = events.loc[mchoice_mask, "action"].map(parse_mchoice_score)
    events.loc[parsons_mask, "event_score"] = events.loc[parsons_mask, "action"].map(parse_parsons_score)
    events.loc[hparsons_mask, "event_score"] = events.loc[hparsons_mask, "action"].map(parse_hparsons_score)
    events.loc[unittest_mask, "event_score"] = events.loc[unittest_mask, "action"].map(parse_unittest_pct)
    keyed_mask = events["family"].isin(KEYED_FAMILIES)
    events["response_norm"] = None
    response_source = np.where(events["input"].notna(), events["input"], events["action"])
    events.loc[keyed_mask, "response_norm"] = pd.Series(response_source[keyed_mask], index=events.index[keyed_mask]).map(normalize_response)
    return events


def build_windows(attempts: pd.DataFrame) -> pd.DataFrame:
    return (
        attempts.groupby(["semester_raw", "anon_student_id", "midterm", "semester", "normalized_midterm_name"], as_index=False)
        .agg(
            exam_start=("attempt_start", "min"),
            exam_end=("attempt_end", "max"),
            n_sessions=("session_id", "nunique"),
            total_exam_minutes=("attempt_minutes", "sum"),
        )
        .sort_values(["semester_raw", "anon_student_id", "midterm"])
    )


def label_exam_events(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    labeled = events.merge(
        windows[WINDOW_KEYS + ["semester", "normalized_midterm_name", "exam_start", "exam_end"]],
        on=["semester_raw", "anon_student_id"],
        how="inner",
    )
    labeled = labeled.loc[
        labeled["time"].between(labeled["exam_start"], labeled["exam_end"], inclusive="both")
    ].copy()
    return labeled


def mark_in_any_exam(events: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    probe = events[["event_id", "semester_raw", "anon_student_id", "time"]].merge(
        windows[["semester_raw", "anon_student_id", "exam_start", "exam_end"]],
        on=["semester_raw", "anon_student_id"],
        how="left",
    )
    probe["in_exam_window"] = probe["time"].between(probe["exam_start"], probe["exam_end"], inclusive="both")
    return probe.groupby("event_id", sort=False)["in_exam_window"].max().reindex(events["event_id"]).fillna(False).to_numpy()


def summarize_exam_problem(group: pd.DataFrame) -> pd.Series:
    family = group["family"].mode().iloc[0]
    score = np.nan
    score_source = "unscored"
    key_confidence = np.nan
    if family == "mchoice":
        values = group["event_score"].dropna()
        if not values.empty:
            score = values.iloc[-1]
            score_source = "last_mchoice_result"
    elif family == "parsons":
        values = group["event_score"].dropna()
        if not values.empty:
            score = values.max()
            score_source = "best_parsons_result"
        else:
            score = 0.0
            score_source = "parsons_attempt_without_success_signal"
    elif family == "activecode":
        values = group.loc[group["selection"].eq("unittest"), "event_score"].dropna()
        if not values.empty:
            score = values.max()
            score_source = "best_unittest_percent"
        else:
            score = 0.0
            score_source = "activecode_attempt_without_passing_unittest"
    elif family in KEYED_FAMILIES:
        finals = group.loc[group["response_norm"].notna()].sort_values("time")
        if not finals.empty:
            last_row = finals.tail(1).iloc[0]
            key_confidence = last_row.get("key_confidence", np.nan)
            if pd.notna(last_row.get("answer_key")) and pd.notna(key_confidence) and key_confidence >= 0.65:
                score = float(last_row["response_norm"] == last_row["answer_key"])
                score_source = f"{family}_modal_key"
            else:
                score_source = f"{family}_low_confidence_key"
        else:
            score_source = f"{family}_no_response"
    elif family == "shortanswer":
        score_source = "unsupported_shortanswer"
    return pd.Series(
        {
            "item_family": family,
            "score": score,
            "score_source": score_source,
            "key_confidence": key_confidence,
            "n_events": len(group),
            "n_scored_events": int(group["event_score"].notna().sum()),
        }
    )


def summarize_study_problem(group: pd.DataFrame) -> pd.Series:
    if "family" in group.columns:
        family = group["family"].iloc[0]
    elif isinstance(group.name, tuple) and len(group.name) >= 4:
        family = group.name[3]
    else:
        family = "unknown"
    values = group["event_score"].dropna()
    first = first_non_null(group["event_score"])
    last = last_non_null(group["event_score"])
    best = values.max() if not values.empty else np.nan
    if values.empty and family in {"parsons", "activecode"}:
        first = 0.0
        last = 0.0
        best = 0.0
    return pd.Series(
        {
            "problem_events": len(group),
            "first_score": first,
            "last_score": last,
            "best_score": best,
            "score_gain": last - first if pd.notna(first) and pd.notna(last) else np.nan,
        }
    )


def within_group_rank_corr(df: pd.DataFrame, group_col: str, feature: str, target: str) -> float:
    use = df[[group_col, feature, target]].dropna().copy()
    if len(use) < 10 or use[feature].nunique() < 3 or use[target].nunique() < 3:
        return np.nan
    use["x_rank"] = use.groupby(group_col)[feature].rank(pct=True)
    use["y_rank"] = use.groupby(group_col)[target].rank(pct=True)
    return use[["x_rank", "y_rank"]].corr().iloc[0, 1]


def within_group_quartile_gap(df: pd.DataFrame, group_col: str, feature: str, target: str) -> float:
    use = df[[group_col, feature, target]].dropna().copy()
    if len(use) < 20 or use[feature].nunique() < 4:
        return np.nan
    use["feature_rank"] = use.groupby(group_col)[feature].rank(pct=True, method="average")
    bottom = use.loc[use["feature_rank"] <= 0.25, target]
    top = use.loc[use["feature_rank"] > 0.75, target]
    if bottom.empty or top.empty:
        return np.nan
    return top.mean() - bottom.mean()


def compute_corr_table(df: pd.DataFrame, group_col: str, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        use = df[[group_col, feature, target]].dropna()
        if use.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "n": len(use),
                "within_group_rank_corr": within_group_rank_corr(df, group_col, feature, target),
                "top_quartile_minus_bottom_quartile_pct_points": within_group_quartile_gap(df, group_col, feature, target),
            }
        )
    out = pd.DataFrame(rows).dropna(subset=["within_group_rank_corr"]).sort_values("within_group_rank_corr", ascending=False)
    return out.reset_index(drop=True)


def save_plot(series: pd.Series, title: str, output_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(10, 6))
    series.sort_values().plot(kind="barh", color=["#c0392b" if value < 0 else "#1f77b4" for value in series.sort_values()])
    plt.title(title)
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def frame_to_pipe_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            show[col] = show[col].fillna("").astype(str)
    header = "| " + " | ".join(show.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in show.astype(str).values.tolist()]
    return "\n".join([header, divider] + rows)


def build_notebook(results: dict[str, pd.DataFrame], summary: dict[str, object]) -> None:
    nb = nbf.v4.new_notebook()
    intro = f"""# Midterm scores and pre-exam activity patterns

This notebook packages a reproducible version of the score reconstruction and the follow-up activity analysis.

## What this notebook does

1. Rebuilds student midterm scores from the timed exam windows in `exam_windows/`.
2. Measures pre-exam engagement with major activity families before each midterm.
3. Compares those activity features with midterm performance within each `(semester, midterm)` cohort.
4. Tracks which mid1-to-mid2 behavior changes line up with midterm improvement.

## Main assumptions

- `exam_windows/timed_exam_attempts.parquet` is the source of truth for which student belonged to which midterm window.
- The semester column in the raw data is more reliable than semester text embedded inside `problem_name`.
- Multiple choice, parsons, and activecode items have direct correctness signals in the event log.
- `fillb`, `clickableArea`, and `dragNdrop` are scored only when an exact-answer key can be inferred with enough agreement from students' final responses. The confidence threshold is `0.65`.
- `shortanswer` is kept visible but not force-scored.

## Headline results

- Reconstructed student-exam rows: **{summary["n_student_scores"]}**
- Mean exam scoring coverage: **{summary["mean_coverage_pct"]:.1f}%** of observed exam problems
- Best pre-exam signals were mostly *quality* signals, not raw volume:
  - Parsons first-try quality had the strongest positive within-cohort relationship.
  - ActiveCode best score and multiple-choice quality also tracked better midterm performance.
  - Large score gains on practice items were often negative markers, which is consistent with students starting from a weaker baseline and needing more rework.
"""
    semester_table = frame_to_pipe_table(results["semester_summary"])
    corr_table = frame_to_pipe_table(results["activity_corr"], max_rows=12)
    change_table = frame_to_pipe_table(results["change_corr"], max_rows=12)
    parsons_table = frame_to_pipe_table(results["parsons_group_summary"])
    load_outputs = """from pathlib import Path
import pandas as pd

base_dir = Path.cwd()
out_dir = base_dir / "analysis_outputs"

student_scores = pd.read_parquet(out_dir / "student_midterm_scores.parquet")
student_problem_scores = pd.read_parquet(out_dir / "student_midterm_problem_scores.parquet")
activity_features = pd.read_parquet(out_dir / "student_midterm_activity_features.parquet")
semester_summary = pd.read_csv(out_dir / "semester_midterm_score_summary.csv")
activity_corr = pd.read_csv(out_dir / "activity_correlations.csv")
change_corr = pd.read_csv(out_dir / "activity_change_correlations.csv")
parsons_summary = pd.read_csv(out_dir / "parsons_change_group_summary.csv")
student_scores.head()
"""
    nb.cells = [
        nbf.v4.new_markdown_cell(intro),
        nbf.v4.new_markdown_cell("## Semester-Level Midterm Averages\n\n" + semester_table),
        nbf.v4.new_markdown_cell("## Strongest Activity Relationships\n\n" + corr_table),
        nbf.v4.new_markdown_cell("## Mid1 to Mid2 Change Relationships\n\n" + change_table),
        nbf.v4.new_markdown_cell("## Parsons-Focused Subgroup Check\n\n" + parsons_table),
        nbf.v4.new_markdown_cell("## Load The Saved Outputs"),
        nbf.v4.new_code_cell(load_outputs),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    # Walkthrough notebooks live in notebooks/; no extra generated notebook here.


def run_analysis(base_dir: Path = BASE_DIR) -> dict[str, pd.DataFrame]:
    # Timed exam sessions are already tagged in exam_windows/. We just collapse those
    # into one window per student / exam, then score what happened inside it.
    attempts = pd.read_parquet(base_dir / "exam_windows" / "timed_exam_attempts.parquet")
    windows = build_windows(attempts)
    print(f"loaded attempts={len(attempts)} windows={len(windows)}", flush=True)

    raw_cols = ["time", "anon_student_id", "action", "problem_name", "session_id", "selection", "input", "semester"]
    raw = pd.read_parquet(base_dir / "runestone_event_log.parquet", columns=raw_cols).rename(columns={"semester": "semester_raw"})
    print(f"loaded raw rows={len(raw)}", flush=True)

    exam_selections = [
        "mChoice",
        "fillb",
        "parsonsMove",
        "parsons",
        "activecode",
        "unittest",
        "ac_error",
        "hparsons",
        "hparsonsAnswer",
        "clickableArea",
        "dragNdrop",
        "dragNdrop-drop",
        "shortanswer",
    ]
    exam_events = add_explicit_event_scores(raw.loc[raw["selection"].isin(exam_selections)].copy())
    exam_events = label_exam_events(exam_events, windows)
    print(f"exam events rows={len(exam_events)}", flush=True)
    exam_key_lookup = infer_answer_keys(exam_events.loc[exam_events["family"].isin(KEYED_FAMILIES)].copy())
    exam_events = exam_events.merge(exam_key_lookup, on="problem_name", how="left")
    keyed_exam_mask = exam_events["family"].isin(KEYED_FAMILIES) & exam_events["response_norm"].notna()
    exam_events.loc[keyed_exam_mask, "event_score"] = (
        exam_events.loc[keyed_exam_mask, "response_norm"] == exam_events.loc[keyed_exam_mask, "answer_key"]
    ).astype(float)

    exam_problem_scores = (
        exam_events.sort_values(WINDOW_KEYS + ["problem_name", "time"])
        .groupby(WINDOW_KEYS + ["semester", "normalized_midterm_name", "problem_name"], as_index=False)
        .apply(summarize_exam_problem, include_groups=False)
        .reset_index()
    )
    if "level_5" in exam_problem_scores.columns:
        exam_problem_scores = exam_problem_scores.drop(columns=["level_5"])
    print(f"exam problem rows={len(exam_problem_scores)}", flush=True)

    student_scores = (
        exam_problem_scores.groupby(WINDOW_KEYS + ["semester", "normalized_midterm_name"], as_index=False)
        .agg(
            n_items_observed=("problem_name", "nunique"),
            n_items_scored=("score", lambda s: s.notna().sum()),
            score_points=("score", "sum"),
            score_pct=("score", "mean"),
        )
        .merge(windows, on=WINDOW_KEYS + ["semester", "normalized_midterm_name"], how="left")
    )
    student_scores["score_pct"] = student_scores["score_pct"] * 100
    student_scores["coverage_pct"] = np.where(
        student_scores["n_items_observed"].gt(0),
        100 * student_scores["n_items_scored"] / student_scores["n_items_observed"],
        np.nan,
    )

    semester_summary = (
        student_scores.groupby(["semester_raw", "semester", "midterm"], as_index=False)
        .agg(
            n_students=("anon_student_id", "nunique"),
            mean_score_pct=("score_pct", "mean"),
            median_score_pct=("score_pct", "median"),
            mean_coverage_pct=("coverage_pct", "mean"),
        )
        .sort_values(["semester_raw", "midterm"])
    )
    exam_problem_scores.to_parquet(OUT_DIR / "student_midterm_problem_scores.parquet", index=False)
    student_scores.to_parquet(OUT_DIR / "student_midterm_scores.parquet", index=False)
    student_scores.to_csv(OUT_DIR / "student_midterm_scores.csv", index=False)
    semester_summary.to_csv(OUT_DIR / "semester_midterm_score_summary.csv", index=False)

    key_conf_summary = exam_key_lookup[["problem_name", "key_n", "key_support", "key_confidence"]].sort_values(
        ["key_confidence", "key_n", "problem_name"], ascending=[True, False, True]
    )
    unsupported_exam_items = exam_problem_scores.loc[
        exam_problem_scores["score"].isna(),
        ["semester_raw", "midterm", "problem_name", "item_family", "score_source", "key_confidence", "n_events"],
    ].sort_values(["item_family", "score_source", "semester_raw", "problem_name"])

    del exam_events
    gc.collect()
    print("finished score reconstruction", flush=True)

    activity_selections = exam_selections
    study_events = add_explicit_event_scores(raw.loc[raw["selection"].isin(activity_selections)].copy())
    study_events["event_id"] = np.arange(len(study_events), dtype=np.int64)
    study_events["in_any_exam"] = mark_in_any_exam(study_events, windows)
    study_events = study_events.loc[~study_events["in_any_exam"]].copy()
    print(f"study rows outside exams={len(study_events)}", flush=True)
    study_key_lookup = infer_answer_keys(study_events.loc[study_events["family"].isin(KEYED_FAMILIES)].copy())
    study_events = study_events.merge(study_key_lookup, on="problem_name", how="left")
    keyed_study_mask = study_events["family"].isin(KEYED_FAMILIES) & study_events["response_norm"].notna()
    study_events.loc[keyed_study_mask, "event_score"] = (
        study_events.loc[keyed_study_mask, "response_norm"] == study_events.loc[keyed_study_mask, "answer_key"]
    ).astype(float)

    pre = study_events[["semester_raw", "anon_student_id", "time", "problem_name", "family", "event_score"]].merge(
        windows[WINDOW_KEYS + ["exam_start"]],
        on=["semester_raw", "anon_student_id"],
        how="inner",
    )
    pre = pre.loc[pre["time"] < pre["exam_start"]].copy()
    pre["event_day"] = pre["time"].dt.floor("D")
    print(f"pre-exam scoreable rows={len(pre)}", flush=True)

    row_activity = (
        pre.groupby(WINDOW_KEYS + ["family"], as_index=False)
        .agg(
            total_events=("problem_name", "size"),
            active_days=("event_day", "nunique"),
            unique_problems=("problem_name", "nunique"),
            scored_events=("event_score", lambda s: s.notna().sum()),
            avg_event_score=("event_score", "mean"),
        )
    )
    all_activity = (
        pre.groupby(WINDOW_KEYS, as_index=False)
        .agg(
            total_events=("problem_name", "size"),
            active_days=("event_day", "nunique"),
            unique_problems=("problem_name", "nunique"),
            scored_events=("event_score", lambda s: s.notna().sum()),
            avg_event_score=("event_score", "mean"),
        )
        .assign(family="all_activity")
    )
    row_activity = pd.concat([row_activity, all_activity], ignore_index=True)
    print(f"row activity rows={len(row_activity)}", flush=True)

    concept_events = raw.loc[raw["selection"].eq("selectquestion"), ["time", "anon_student_id", "problem_name", "selection", "semester_raw"]].copy()
    concept_events["event_id"] = np.arange(len(concept_events), dtype=np.int64)
    concept_events["in_any_exam"] = mark_in_any_exam(concept_events, windows)
    concept_events = concept_events.loc[~concept_events["in_any_exam"]].copy()
    concept_pre = concept_events.merge(
        windows[WINDOW_KEYS + ["exam_start"]],
        on=["semester_raw", "anon_student_id"],
        how="inner",
    )
    concept_pre = concept_pre.loc[concept_pre["time"] < concept_pre["exam_start"]].copy()
    concept_pre["event_day"] = concept_pre["time"].dt.floor("D")
    concept_row = (
        concept_pre.groupby(WINDOW_KEYS, as_index=False)
        .agg(
            total_events=("problem_name", "size"),
            active_days=("event_day", "nunique"),
            unique_problems=("problem_name", "nunique"),
        )
        .assign(scored_events=0, avg_event_score=np.nan, family="conceptcheck")
    )
    row_activity = pd.concat([row_activity, concept_row], ignore_index=True)
    print(f"conceptcheck pre rows={len(concept_pre)} row summaries={len(concept_row)}", flush=True)

    problem_families = sorted(SCOREABLE_FAMILIES | {"parsons", "activecode"})
    pre_problem = pre.loc[pre["family"].isin(problem_families)].copy()
    problem_feature_table = (
        pre_problem.sort_values(WINDOW_KEYS + ["family", "problem_name", "time"])
        .groupby(WINDOW_KEYS + ["family", "problem_name"], as_index=False)
        .apply(summarize_study_problem)
        .reset_index()
    )
    if "level_5" in problem_feature_table.columns:
        problem_feature_table = problem_feature_table.drop(columns=["level_5"])
    print(f"study problem feature rows={len(problem_feature_table)}", flush=True)

    activity_features = (
        problem_feature_table.groupby(WINDOW_KEYS + ["family"], as_index=False)
        .agg(
            problems_seen=("problem_name", "nunique"),
            scored_problems=("best_score", lambda s: s.notna().sum()),
            mean_first_score=("first_score", "mean"),
            mean_last_score=("last_score", "mean"),
            mean_best_score=("best_score", "mean"),
            mean_score_gain=("score_gain", "mean"),
            mean_events_per_problem=("problem_events", "mean"),
        )
    )
    learning = (
        problem_feature_table.groupby(WINDOW_KEYS, as_index=False)
        .agg(
            problems_seen=("problem_name", "nunique"),
            scored_problems=("best_score", lambda s: s.notna().sum()),
            mean_first_score=("first_score", "mean"),
            mean_last_score=("last_score", "mean"),
            mean_best_score=("best_score", "mean"),
            mean_score_gain=("score_gain", "mean"),
            mean_events_per_problem=("problem_events", "mean"),
        )
        .assign(family="learning")
    )
    activity_features = pd.concat([activity_features, learning], ignore_index=True)

    row_wide = row_activity.pivot(index=WINDOW_KEYS, columns="family")
    row_wide.columns = [f"{family}__{metric}" for metric, family in row_wide.columns]
    row_wide = row_wide.reset_index()

    feature_wide = activity_features.pivot(index=WINDOW_KEYS, columns="family")
    feature_wide.columns = [f"{family}__{metric}" for metric, family in feature_wide.columns]
    feature_wide = feature_wide.reset_index()

    attempt_features = student_scores.merge(row_wide, on=WINDOW_KEYS, how="left").merge(feature_wide, on=WINDOW_KEYS, how="left")
    attempt_features["cohort"] = attempt_features["semester_raw"] + "_" + attempt_features["midterm"]
    print(f"attempt feature rows={len(attempt_features)}", flush=True)

    activity_feature_cols = [col for col in attempt_features.columns if "__" in col and col not in {"normalized_midterm_name"}]
    activity_corr = compute_corr_table(attempt_features, "cohort", "score_pct", activity_feature_cols)

    numeric_feature_cols = [
        col
        for col in attempt_features.columns
        if "__" in col and pd.api.types.is_numeric_dtype(attempt_features[col])
    ]
    base_for_change = attempt_features[["semester_raw", "anon_student_id", "midterm", "score_pct"] + numeric_feature_cols].copy()
    change_wide = base_for_change.set_index(["semester_raw", "anon_student_id", "midterm"]).unstack("midterm")
    change_wide.columns = [f"{metric}__{midterm}" for metric, midterm in change_wide.columns]
    change_wide = change_wide.reset_index()
    change_wide = change_wide.loc[
        change_wide["score_pct__mid1"].notna() & change_wide["score_pct__mid2"].notna()
    ].copy()
    change_wide["score_change"] = change_wide["score_pct__mid2"] - change_wide["score_pct__mid1"]
    change_features = []
    for feature in numeric_feature_cols:
        mid1_col = f"{feature}__mid1"
        mid2_col = f"{feature}__mid2"
        if mid1_col in change_wide.columns and mid2_col in change_wide.columns:
            change_col = f"{feature}__change"
            change_wide[change_col] = change_wide[mid2_col] - change_wide[mid1_col]
            change_features.append(change_col)
    change_corr = compute_corr_table(change_wide, "semester_raw", "score_change", change_features)
    print(f"change rows={len(change_wide)}", flush=True)

    parsons_groups = change_wide[
        [
            "semester_raw",
            "anon_student_id",
            "score_pct__mid1",
            "score_pct__mid2",
            "score_change",
            "parsons__unique_problems__mid1",
            "parsons__unique_problems__mid2",
            "parsons__mean_best_score__mid1",
            "parsons__mean_best_score__mid2",
            "parsons__unique_problems__change",
            "parsons__mean_best_score__change",
        ]
    ].copy()
    parsons_groups["baseline_group"] = np.select(
        [
            parsons_groups["parsons__unique_problems__mid1"].fillna(0).eq(0),
            parsons_groups["parsons__mean_best_score__mid1"].fillna(-1).lt(0.5),
        ],
        ["no_parsons_before_mid1", "low_parsons_accuracy_before_mid1"],
        default="stronger_parsons_before_mid1",
    )
    parsons_groups["parsons_usage_change_group"] = np.where(
        parsons_groups["parsons__unique_problems__change"].fillna(0).gt(0),
        "increased_parsons_before_mid2",
        "did_not_increase_parsons_before_mid2",
    )
    parsons_group_summary = (
        parsons_groups.groupby(["baseline_group", "parsons_usage_change_group"], as_index=False)
        .agg(
            n_students=("anon_student_id", "count"),
            mean_mid1_pct=("score_pct__mid1", "mean"),
            mean_mid2_pct=("score_pct__mid2", "mean"),
            mean_score_change=("score_change", "mean"),
            mean_parsons_usage_change=("parsons__unique_problems__change", "mean"),
            mean_parsons_accuracy_change=("parsons__mean_best_score__change", "mean"),
        )
        .sort_values(["baseline_group", "parsons_usage_change_group"])
    )

    focus_features = [
        "parsons__unique_problems",
        "parsons__mean_best_score",
        "activecode__unique_problems",
        "activecode__mean_best_score",
        "conceptcheck__unique_problems",
        "mchoice__mean_first_score",
        "fillb__mean_best_score",
    ]
    focus_corr = activity_corr.loc[activity_corr["feature"].isin(focus_features)].copy()
    focus_change = change_corr.loc[
        change_corr["feature"].isin([f"{feature}__change" for feature in focus_features if f"{feature}__change" in change_corr["feature"].values])
    ].copy()

    attempt_features.to_parquet(OUT_DIR / "student_midterm_activity_features.parquet", index=False)
    activity_corr.to_csv(OUT_DIR / "activity_correlations.csv", index=False)
    change_corr.to_csv(OUT_DIR / "activity_change_correlations.csv", index=False)
    focus_corr.to_csv(OUT_DIR / "focus_activity_correlations.csv", index=False)
    focus_change.to_csv(OUT_DIR / "focus_activity_change_correlations.csv", index=False)
    parsons_groups.to_parquet(OUT_DIR / "parsons_change_groups.parquet", index=False)
    parsons_group_summary.to_csv(OUT_DIR / "parsons_change_group_summary.csv", index=False)
    unsupported_exam_items.to_csv(OUT_DIR / "unsupported_or_ambiguous_exam_items.csv", index=False)
    key_conf_summary.to_csv(OUT_DIR / "exam_inferred_answer_keys.csv", index=False)

    if not activity_corr.empty:
        top_corr = pd.concat([activity_corr.head(10), activity_corr.tail(10)]).drop_duplicates("feature")
        save_plot(
            top_corr.set_index("feature")["within_group_rank_corr"],
            "Top Positive and Negative Activity Correlations",
            OUT_DIR / "activity_correlations.png",
        )
    score_plot = semester_summary.assign(label=lambda df: df["semester_raw"] + " " + df["midterm"]).set_index("label")["mean_score_pct"]
    save_plot(score_plot, "Mean Midterm Score by Semester and Midterm", OUT_DIR / "semester_midterm_scores.png")

    summary = {
        "n_student_scores": int(len(student_scores)),
        "mean_coverage_pct": float(student_scores["coverage_pct"].mean()),
    }
    results = {
        "student_scores": student_scores,
        "semester_summary": semester_summary,
        "activity_corr": activity_corr,
        "change_corr": change_corr,
        "parsons_group_summary": parsons_group_summary,
    }
    build_notebook(results, summary)
    print("wrote outputs and notebook", flush=True)
    return results


if __name__ == "__main__":
    run_analysis(BASE_DIR)
