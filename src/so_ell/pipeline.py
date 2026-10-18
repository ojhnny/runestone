"""The pipeline, as plain functions. One function per stage, each reads the
previous stage's files from disk and writes its own, so any stage can be
re-run on its own (the CLI and Dagster both call these).

    ingest -> exam_windows -> scores -> features -> analysis -> models -> report
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from so_ell import config, exam_windows, ingest, models, sql_features, stats, validation
from so_ell.config import WINDOW_KEYS
from so_ell.features import build_practice_features, mid1_to_mid2_changes, score_study_events
from so_ell.scores import build_windows, load_scoreable_events, score_exams

P = config.PROCESSED_DIR
T = config.TABLES_DIR

SCORES = P / "student_exam_scores.parquet"
PROBLEM_SCORES = P / "student_exam_problem_scores.parquet"
FEATURES = P / "student_exam_features.parquet"
LEAD_TIME_CUTOFFS = [28, 21, 14, 7, 3, 0]

HEADLINE_FEATURES = [
    "parsons__mean_first_score",
    "parsons__avg_event_score",
    "mchoice__mean_first_score",
    "activecode__mean_best_score",
    "coverage__unique_chapters",
    "coverage__chapter_diversity",
    "timing__share_last_7d",
    "consistency__span_days",
    "consistency__problems_per_active_day",
    "mix__coding_share",
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _csv(df: pd.DataFrame, name: str, directory: Path = T) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / name, index=False)


# ---------------------------------------------------------------- stage 1
def stage_ingest(verify_checksums: bool = True) -> None:
    config.ensure_dirs()
    _log("ingest: reading raw Datashop exports")
    counts = ingest.run_ingest(verify_checksums=verify_checksums)
    _csv(counts, "ingest_row_counts.csv", config.QUALITY_DIR)
    _log(f"ingest: wrote {counts['rows'].sum():,} rows; running event log checks")
    _csv(validation.event_log_checks(), "event_log_checks.csv", config.QUALITY_DIR)


# ---------------------------------------------------------------- stage 2
def stage_exam_windows() -> None:
    config.ensure_dirs()
    result = exam_windows.run_exam_windows()
    validation.validate(result["attempts"], validation.exam_attempts_schema)
    if not result["missing"].empty:
        raise ValueError(f"No timed midterm found for: {result['missing'].to_dict('records')}")
    _csv(result["summary"], "exam_window_summary.csv")
    _csv(result["name_audit"], "exam_name_audit.csv")
    _log(f"exam_windows: {len(result['attempts']):,} attempts across {len(result['summary'])} exams")


# ---------------------------------------------------------------- stage 3
def _windows() -> pd.DataFrame:
    return build_windows(pd.read_parquet(config.EXAM_ATTEMPTS_PARQUET))


def stage_scores() -> None:
    config.ensure_dirs()
    windows = _windows()
    raw = load_scoreable_events()
    _log(f"scores: {len(raw):,} answer events loaded")
    result = score_exams(raw, windows)
    validation.validate(result["student_scores"], validation.student_scores_schema)
    result["problem_scores"].to_parquet(PROBLEM_SCORES, index=False)
    result["student_scores"].to_parquet(SCORES, index=False)
    _csv(result["semester_summary"], "cohort_score_summary.csv")
    _csv(result["answer_keys"].drop(columns=["answer_key"]), "inferred_answer_keys.csv")
    _csv(result["unsupported_summary"], "unscored_items_summary.csv")
    _log(f"scores: {len(result['student_scores']):,} student-exam scores")


# ---------------------------------------------------------------- stage 4
def _features_at_cutoff(study, windows, scores, cutoff: int) -> pd.DataFrame:
    practice = build_practice_features(study, windows, scores, cutoff_days=cutoff)
    sql = sql_features.build_sql_features(cutoff_days=cutoff)
    keys = WINDOW_KEYS
    extra = [c for c in sql["student_features"].columns if c not in keys]
    merged = practice.merge(sql["student_features"][keys + extra], on=keys, how="left")
    return merged, sql


def stage_features() -> None:
    config.ensure_dirs()
    windows = _windows()
    scores = pd.read_parquet(SCORES)
    raw = load_scoreable_events()
    study = score_study_events(raw, windows)
    del raw
    _log(f"features: {len(study):,} practice events outside exam windows")

    features, sql = _features_at_cutoff(study, windows, scores, 0)
    validation.validate(features, validation.features_schema)
    features.to_parquet(FEATURES, index=False)
    _csv(sql["overview"], "event_log_overview.csv")
    _csv(sql["by_semester"], "events_by_semester.csv")
    _csv(sql["chapter_cohort"], "chapter_activity_by_cohort.csv")
    timing = public_timing_table(sql["timing_bins"], features)
    _csv(timing, "practice_timing_by_score_band.csv")

    for cutoff in LEAD_TIME_CUTOFFS:
        if cutoff == 0:
            continue
        early, _ = _features_at_cutoff(study, windows, scores, cutoff)
        early.to_parquet(P / f"student_exam_features_cutoff_{cutoff}d.parquet", index=False)
        _log(f"features: cutoff {cutoff}d done")
    _log(f"features: {len(features):,} rows x {features.shape[1]} columns")


def public_timing_table(timing_bins: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Share of last-28-day practice in each timing bin, by score quartile (no student ids)."""
    scores = stats.add_cohort(features)[WINDOW_KEYS + ["cohort", "score_pct"]].copy()
    scores["score_band"] = scores.groupby("cohort")["score_pct"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    )
    merged = timing_bins.merge(scores, on=WINDOW_KEYS, how="inner")
    merged["share_pct"] = 100 * merged["n_events"] / merged.groupby(WINDOW_KEYS)["n_events"].transform("sum")
    return (
        merged.groupby(["semester_raw", "midterm", "score_band", "timing_bin"], observed=False)
        .agg(mean_share_pct=("share_pct", "mean"), n_students=("anon_student_id", "nunique"))
        .reset_index()
    )


# ---------------------------------------------------------------- stage 5
def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "__" in c and pd.api.types.is_numeric_dtype(df[c])]


def stage_analysis(n_boot: int = 1000) -> None:
    config.ensure_dirs()
    features = stats.add_cohort(pd.read_parquet(FEATURES))
    corr = stats.compute_corr_table(features, "cohort", "score_pct", numeric_feature_cols(features))
    _csv(corr, "feature_signals.csv")
    models_tbl = stats.fit_parsons_models(features)
    _csv(models_tbl, "parsons_models.csv")
    boot = stats.bootstrap_quartile_gap(features, n_boot=n_boot)
    _csv(pd.DataFrame([boot]), "parsons_gap_bootstrap.csv")
    quartiles = pd.concat([stats.quartile_table(features, c) for c in HEADLINE_FEATURES if c in features], ignore_index=True)
    _csv(quartiles, "feature_quartile_scores.csv")

    change = mid1_to_mid2_changes(features)
    change_cols = [c for c in change.columns if c.endswith("__change")]
    _csv(stats.compute_corr_table(change, "semester_raw", "score_change", change_cols), "change_signals.csv")
    _csv(parsons_change_groups(change), "parsons_change_groups.csv")

    _csv(key_threshold_sensitivity(features), "key_threshold_sensitivity.csv")
    _log("analysis: correlations, fixed-effects models, bootstrap and sensitivity written")


def parsons_change_groups(change: pd.DataFrame, min_group: int = 10) -> pd.DataFrame:
    """Students split by Parsons first-try success before mid1 (above/below their
    semester's median) and by whether they did new Parsons problems before mid2.

    Groups smaller than ``min_group`` students are suppressed so no published
    row describes an individual.
    """
    use = change.copy()
    first = use["parsons__mean_first_score__mid1"]
    median = first.groupby(use["semester_raw"]).transform("median")
    use["baseline_group"] = "no_parsons_before_mid1"
    use.loc[first.notna() & first.lt(median), "baseline_group"] = "below_median_first_try_before_mid1"
    use.loc[first.notna() & first.ge(median), "baseline_group"] = "above_median_first_try_before_mid1"
    # Features are cumulative, so "change" is the practice done between the exams.
    use["usage_group"] = use["parsons__unique_problems__change"].fillna(0).gt(0).map(
        {True: "did_more_parsons_before_mid2", False: "no_new_parsons_before_mid2"}
    )
    out = (
        use.groupby(["baseline_group", "usage_group"], as_index=False)
        .agg(
            n_students=("anon_student_id", "count"),
            mean_mid1_pct=("score_pct__mid1", "mean"),
            mean_mid2_pct=("score_pct__mid2", "mean"),
            mean_score_change=("score_change", "mean"),
        )
    )
    return out.loc[out["n_students"] >= min_group].reset_index(drop=True)


def key_threshold_sensitivity(features: pd.DataFrame) -> pd.DataFrame:
    """Does the headline survive a different answer-key threshold, or none at all?"""
    windows = _windows()
    raw = load_scoreable_events()
    rows = []
    for threshold in [0.55, 0.65, 0.75, 0.85, 1.01]:
        scores = score_exams(raw, windows, key_threshold=threshold)["student_scores"]
        swapped = features.drop(columns=["score_pct"]).merge(scores[WINDOW_KEYS + ["score_pct"]], on=WINDOW_KEYS)
        fe = stats.fit_parsons_models(swapped).iloc[0]
        rows.append(
            {
                "key_threshold": threshold,
                "label": "inferred keys dropped" if threshold > 1 else f"keys need {threshold:.0%} agreement",
                "n_student_exams": len(swapped),
                "parsons_quartile_gap": stats.within_group_quartile_gap(swapped, "cohort", stats.PARSONS, "score_pct"),
                "parsons_rank_corr": stats.within_group_rank_corr(swapped, "cohort", stats.PARSONS, "score_pct"),
                "fe_coef": fe["coef"],
                "fe_ci_low": fe["ci_low"],
                "fe_ci_high": fe["ci_high"],
            }
        )
        _log(f"analysis: key threshold {threshold} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage 6
def stage_models() -> None:
    config.ensure_dirs()
    features = pd.read_parquet(FEATURES)
    results = models.run_model_suite(features)
    comparison = pd.DataFrame([{"model": k} | r.summary for k, r in results.items()])
    _csv(comparison, "at_risk_model_comparison.csv")
    _csv(pd.concat([r.folds for r in results.values()], ignore_index=True), "at_risk_folds.csv")
    _csv(results["gradient_boosting"].importance, "at_risk_importance_gbm.csv")
    _csv(results["logistic"].importance, "at_risk_coefficients_logistic.csv")
    oof = pd.concat([results["logistic"].oof, results["gradient_boosting"].oof], ignore_index=True)
    _csv(models.calibration_table(oof), "at_risk_calibration.csv")

    by_cutoff = {0: features}
    for cutoff in LEAD_TIME_CUTOFFS:
        path = P / f"student_exam_features_cutoff_{cutoff}d.parquet"
        if cutoff and path.exists():
            by_cutoff[cutoff] = pd.read_parquet(path)
    lead = models.lead_time_curve(by_cutoff)
    _csv(lead, "at_risk_lead_time.csv")
    logged = models.log_to_mlflow(results, lead)
    _log(f"models: done ({'logged to MLflow' if logged else 'MLflow not installed, skipped logging'})")


# ---------------------------------------------------------------- stage 7
def stage_report() -> None:
    from so_ell import report

    config.ensure_dirs()
    report.build_all()
    _log("report: figures, headline metrics and Looker tables written")


STAGES = {
    "ingest": stage_ingest,
    "exam-windows": stage_exam_windows,
    "scores": stage_scores,
    "features": stage_features,
    "analysis": stage_analysis,
    "models": stage_models,
    "report": stage_report,
}


def run_all(start: str = "ingest") -> None:
    names = list(STAGES)
    for name in names[names.index(start):]:
        t0 = time.time()
        STAGES[name]()
        _log(f"stage {name} finished in {time.time() - t0:.0f}s")
    (config.OUTPUTS_DIR / "run_info.json").write_text(
        json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": names[names.index(start):]}, indent=2)
    )
