"""Rebuild SQL features, models, and Looker tables from saved score outputs.

By default this does NOT rerun the 20M-row pandas score reconstruction. That
still lives in rebuild_midterm_scores.py and takes a while. If the parquet outputs
are already there, this script is the part you actually rerun day to day.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ebook_analysis.looker_export import export_looker_tables
from ebook_analysis.paths import BASE_DIR, OUT_DIR, ensure_output_dirs
from ebook_analysis.duckdb_queries import build_sql_features
from ebook_analysis.stats import (
    add_cohort,
    at_risk_cv,
    compute_corr_table,
    fit_parsons_models,
    quartile_table,
)

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


def load_score_outputs(out_dir: Path = OUT_DIR) -> dict[str, pd.DataFrame]:
    features_path = out_dir / "student_midterm_activity_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} is missing. Run `python rebuild_midterm_scores.py` once first."
        )
    return {
        "features": pd.read_parquet(features_path),
        "scores": pd.read_parquet(out_dir / "student_midterm_scores.parquet"),
        "semester_summary": pd.read_csv(out_dir / "semester_midterm_score_summary.csv"),
        "activity_corr": pd.read_csv(out_dir / "activity_correlations.csv"),
    }


def merge_sql_features(old: pd.DataFrame, sql_features: pd.DataFrame) -> pd.DataFrame:
    keys = ["semester_raw", "anon_student_id", "midterm"]
    extra_cols = [c for c in sql_features.columns if c not in keys]
    merged = old.merge(sql_features[keys + extra_cols], on=keys, how="left")
    return add_cohort(merged)


def public_timing_table(timing_bins: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    scores = add_cohort(features)[["semester_raw", "anon_student_id", "midterm", "cohort", "score_pct"]].copy()
    scores["score_band"] = scores.groupby("cohort")["score_pct"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    )
    merged = timing_bins.merge(scores, on=["semester_raw", "anon_student_id", "midterm"], how="inner")
    totals = merged.groupby(
        ["semester_raw", "anon_student_id", "midterm"], as_index=False
    )["n_events"].sum().rename(columns={"n_events": "total_events"})
    merged = merged.merge(totals, on=["semester_raw", "anon_student_id", "midterm"], how="left")
    merged["share_pct"] = 100 * merged["n_events"] / merged["total_events"]
    return (
        merged.groupby(["semester_raw", "midterm", "score_band", "timing_bin"], observed=False)
        .agg(mean_share_pct=("share_pct", "mean"), n_students=("anon_student_id", "nunique"))
        .reset_index()
    )


def write_headlines(corr: pd.DataFrame, regression: pd.DataFrame, at_risk: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    parsons = corr.loc[corr["feature"].eq("parsons__mean_first_score")].iloc[0]
    fe = regression.loc[regression["spec"].eq("cohort_fe")].iloc[0]
    controlled = regression.loc[regression["spec"].eq("cohort_fe_plus_effort")].iloc[0]
    oof = at_risk.loc[at_risk["fold"].astype(str).eq("overall_oof")].iloc[0]
    rows = [
        {
            "metric": "parsons_quartile_gap",
            "value": float(parsons["top_quartile_minus_bottom_quartile_pct_points"]),
            "note": "Top vs bottom quartile of Parsons first-try quality, within semester-exam cohort.",
        },
        {
            "metric": "parsons_rank_corr",
            "value": float(parsons["within_group_rank_corr"]),
            "note": "Within-cohort rank correlation, Parsons first-try vs midterm %.",
        },
        {
            "metric": "parsons_fe_coef",
            "value": float(fe["coef"]),
            "note": fe["note"],
        },
        {
            "metric": "parsons_fe_ci_low",
            "value": float(fe["ci_low"]),
            "note": "95% CI lower bound on the cohort FE coefficient.",
        },
        {
            "metric": "parsons_fe_ci_high",
            "value": float(fe["ci_high"]),
            "note": "95% CI upper bound on the cohort FE coefficient.",
        },
        {
            "metric": "parsons_controlled_coef",
            "value": float(controlled["coef"]),
            "note": controlled["note"],
        },
        {
            "metric": "at_risk_oof_auc",
            "value": float(oof["auc"]),
            "note": "Leave-one-semester-out AUC for predicting bottom-quartile exam scores.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "headline_metrics.csv", index=False)
    return out


def run(base_dir: Path = BASE_DIR) -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    out_dir = base_dir / "analysis_outputs"
    print("loading saved score and feature tables", flush=True)
    cached = load_score_outputs(out_dir)

    print("running DuckDB feature queries", flush=True)
    sql = build_sql_features()
    features = merge_sql_features(cached["features"], sql["student_features"])
    print(f"merged feature rows={len(features)} cols={len(features.columns)}", flush=True)

    feature_cols = [
        col
        for col in features.columns
        if "__" in col
        and col != "normalized_midterm_name"
        and pd.api.types.is_numeric_dtype(features[col])
    ]
    activity_corr = compute_corr_table(features, "cohort", "score_pct", feature_cols)

    print("fitting cohort models", flush=True)
    regression = fit_parsons_models(features)
    at_risk_metrics, at_risk_coefs = at_risk_cv(features)

    quartile_frames = [
        quartile_table(features, col)
        for col in HEADLINE_FEATURES
        if col in features.columns
    ]
    quartiles = pd.concat(quartile_frames, ignore_index=True)
    timing_public = public_timing_table(sql["timing_bins"], features)
    headlines = write_headlines(activity_corr, regression, at_risk_metrics, out_dir)

    features.to_parquet(out_dir / "student_exam_features_sql.parquet", index=False)
    sql["student_features"].to_parquet(out_dir / "sql_student_features.parquet", index=False)
    activity_corr.to_csv(out_dir / "activity_correlations_with_sql.csv", index=False)
    regression.to_csv(out_dir / "parsons_regression.csv", index=False)
    at_risk_metrics.to_csv(out_dir / "at_risk_cv.csv", index=False)
    at_risk_coefs.to_csv(out_dir / "at_risk_coefs.csv", index=False)
    quartiles.to_csv(out_dir / "feature_quartile_scores.csv", index=False)
    sql["chapter_cohort"].to_csv(out_dir / "chapter_by_cohort.csv", index=False)
    sql["overview"].to_csv(out_dir / "sql_event_overview.csv", index=False)
    sql["by_semester"].to_csv(out_dir / "sql_events_by_semester.csv", index=False)

    print("writing Looker exports", flush=True)
    export_looker_tables(
        student_features=features,
        activity_corr=activity_corr,
        regression=regression,
        at_risk_metrics=at_risk_metrics,
        at_risk_coefs=at_risk_coefs,
        chapter_cohort=sql["chapter_cohort"],
        timing_public=timing_public,
        quartile_tables=quartiles,
        overview=sql["overview"],
        by_semester=sql["by_semester"],
        semester_summary=cached["semester_summary"],
    )

    print(headlines.to_string(index=False), flush=True)
    return {
        "features": features,
        "activity_corr": activity_corr,
        "regression": regression,
        "at_risk_metrics": at_risk_metrics,
        "headlines": headlines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQL features, stats, and Looker CSVs.")
    parser.add_argument(
        "--rebuild-scores",
        action="store_true",
        help="Also rerun rebuild_midterm_scores.py (slow; 20M-row pandas scoring).",
    )
    args = parser.parse_args()
    if args.rebuild_scores:
        from rebuild_midterm_scores import run_analysis

        print("rebuilding midterm scores from the raw log", flush=True)
        run_analysis(BASE_DIR)
    run(BASE_DIR)


if __name__ == "__main__":
    main()
