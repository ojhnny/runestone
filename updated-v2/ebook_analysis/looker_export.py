"""Write the CSV files you'll actually upload to Looker Studio.

Looker is happier with short, aggregate tables than with 2,000 student rows.
Student-level data (still anonymized) goes under looker_studio/private/ and
stays gitignored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ebook_analysis.paths import LOOKER_DIR, PRIVATE_LOOKER_DIR, ensure_output_dirs


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_looker_tables(
    *,
    student_features: pd.DataFrame,
    activity_corr: pd.DataFrame,
    regression: pd.DataFrame,
    at_risk_metrics: pd.DataFrame,
    at_risk_coefs: pd.DataFrame,
    chapter_cohort: pd.DataFrame,
    timing_public: pd.DataFrame,
    quartile_tables: pd.DataFrame,
    overview: pd.DataFrame,
    by_semester: pd.DataFrame,
    semester_summary: pd.DataFrame,
) -> dict[str, Path]:
    ensure_output_dirs()
    written: dict[str, Path] = {}

    written["overview"] = _write(overview, LOOKER_DIR / "event_log_overview.csv")
    written["by_semester"] = _write(by_semester, LOOKER_DIR / "events_by_semester.csv")
    written["semester_summary"] = _write(
        semester_summary, LOOKER_DIR / "cohort_exam_summary.csv"
    )
    written["signals"] = _write(activity_corr, LOOKER_DIR / "feature_score_signals.csv")
    written["regression"] = _write(regression, LOOKER_DIR / "parsons_fixed_effects.csv")
    written["at_risk_metrics"] = _write(at_risk_metrics, LOOKER_DIR / "at_risk_cross_validation.csv")
    written["at_risk_coefs"] = _write(at_risk_coefs, LOOKER_DIR / "at_risk_coefficients.csv")
    written["chapters"] = _write(chapter_cohort, LOOKER_DIR / "chapter_activity_by_cohort.csv")
    written["timing"] = _write(timing_public, LOOKER_DIR / "practice_timing_by_score_band.csv")
    written["quartiles"] = _write(quartile_tables, LOOKER_DIR / "feature_quartile_exam_scores.csv")

    # Compact cohort KPI table for scorecards.
    kpi = semester_summary.copy()
    kpi["cohort"] = kpi["semester_raw"].astype(str) + "_" + kpi["midterm"].astype(str)
    written["kpis"] = _write(kpi, LOOKER_DIR / "cohort_kpis.csv")

    private_cols = [
        col
        for col in student_features.columns
        if col
        in {
            "semester_raw",
            "anon_student_id",
            "midterm",
            "score_pct",
            "cohort",
            "parsons__mean_first_score",
            "activecode__mean_best_score",
            "mchoice__mean_first_score",
            "all_activity__total_events",
            "all_activity__active_days",
            "coverage__unique_chapters",
            "coverage__chapter_diversity",
            "timing__share_last_7d",
            "mix__coding_share",
        }
        or col.startswith(("coverage__", "timing__", "consistency__", "mix__"))
    ]
    private = student_features.loc[:, ~student_features.columns.duplicated()].copy()
    keep = [c for c in private_cols if c in private.columns]
    written["private_students"] = _write(
        private[keep], PRIVATE_LOOKER_DIR / "student_level_features.csv"
    )
    return written
