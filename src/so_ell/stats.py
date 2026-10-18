"""Cohort-controlled statistics.

Exams differ in difficulty across semesters, so every comparison is made
*within* a cohort = (semester, midterm). Three layers:

1. Descriptive: within-cohort rank correlation and top-vs-bottom quartile gap
   for every feature, with Benjamini-Hochberg false discovery control because
   ~100 features are screened.
2. Inferential: OLS of exam score on Parsons first-try quality with cohort
   fixed effects, standard errors clustered by student (most students sit
   both midterms, so their rows are not independent).
3. Uncertainty on the headline gap: a cluster bootstrap that resamples
   students, not rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

from so_ell.config import RANDOM_SEED

PARSONS = "parsons__mean_first_score"


def add_cohort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cohort"] = out["semester_raw"].astype(str) + "_" + out["midterm"].astype(str)
    return out


def within_group_rank_corr(df: pd.DataFrame, group_col: str, feature: str, target: str) -> float:
    """Pearson correlation of within-cohort percentile ranks (a pooled Spearman)."""
    use = df[[group_col, feature, target]].dropna().copy()
    if len(use) < 10 or use[feature].nunique() < 3 or use[target].nunique() < 3:
        return np.nan
    use["x_rank"] = use.groupby(group_col)[feature].rank(pct=True)
    use["y_rank"] = use.groupby(group_col)[target].rank(pct=True)
    return float(use[["x_rank", "y_rank"]].corr().iloc[0, 1])


def within_group_quartile_gap(df: pd.DataFrame, group_col: str, feature: str, target: str) -> float:
    """Mean target in the top feature quartile minus the bottom one, quartiles cut per cohort."""
    use = df[[group_col, feature, target]].dropna().copy()
    if len(use) < 20 or use[feature].nunique() < 4:
        return np.nan
    use["feature_rank"] = use.groupby(group_col)[feature].rank(pct=True, method="average")
    bottom = use.loc[use["feature_rank"] <= 0.25, target]
    top = use.loc[use["feature_rank"] > 0.75, target]
    if bottom.empty or top.empty:
        return np.nan
    return float(top.mean() - bottom.mean())


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return pd.Series(out, index=p_values.index)


def compute_corr_table(df: pd.DataFrame, group_col: str, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        use = df[[group_col, feature, target]].dropna()
        if use.empty:
            continue
        r = within_group_rank_corr(df, group_col, feature, target)
        n = len(use)
        # t-approximation for a rank correlation; good enough for screening.
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r**2)) if pd.notna(r) else np.nan
        rows.append(
            {
                "feature": feature,
                "n": int(n),
                "within_group_rank_corr": r,
                "p_value": float(2 * sps.t.sf(abs(t), n - 2)) if pd.notna(t) else np.nan,
                "top_quartile_minus_bottom_quartile_pct_points": within_group_quartile_gap(df, group_col, feature, target),
            }
        )
    out = pd.DataFrame(rows).dropna(subset=["within_group_rank_corr"])
    if not out.empty:
        out["q_value_bh"] = benjamini_hochberg(out["p_value"])
    return out.sort_values("within_group_rank_corr", ascending=False).reset_index(drop=True)


def bootstrap_quartile_gap(
    df: pd.DataFrame,
    feature: str = PARSONS,
    target: str = "score_pct",
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """95% CI for the quartile gap, resampling students (clusters) with replacement."""
    use = add_cohort(df)[["cohort", "semester_raw", "anon_student_id", feature, target]].dropna()
    use["cluster"] = use["semester_raw"].astype(str) + "|" + use["anon_student_id"].astype(str)
    use = use.reset_index(drop=True)
    rows_by_cluster = list(use.groupby("cluster").indices.values())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(rows_by_cluster), size=len(rows_by_cluster))
        boot = use.iloc[np.concatenate([rows_by_cluster[i] for i in picks])]
        draws.append(within_group_quartile_gap(boot, "cohort", feature, target))
    draws = np.array([d for d in draws if pd.notna(d)])
    return {
        "estimate": within_group_quartile_gap(use, "cohort", feature, target),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n_boot": len(draws),
    }


def _ols_row(model, spec: str, term: str, n: int, n_clusters: int, note: str) -> dict[str, object]:
    ci = model.conf_int().loc[term]
    return {
        "spec": spec,
        "term": term,
        "n": n,
        "n_student_clusters": n_clusters,
        "coef": float(model.params[term]),
        "std_err": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "r_squared": float(model.rsquared),
        "note": note,
    }


def _prep_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    use = add_cohort(df).rename(
        columns={
            PARSONS: "parsons_first",
            "all_activity__total_events": "total_events",
            "all_activity__active_days": "active_days",
            "coverage__unique_chapters": "unique_chapters",
            "timing__share_last_7d": "share_last_7d",
        }
    )
    use = use.dropna(subset=["score_pct", "parsons_first"]).copy()
    # 0-1 -> percentage points, so the coefficient reads "exam points per Parsons point".
    use["parsons_first"] = use["parsons_first"] * 100
    use["student"] = use["semester_raw"].astype(str) + "|" + use["anon_student_id"].astype(str)
    for col in ["total_events", "active_days", "unique_chapters", "share_last_7d"]:
        if col in use:
            use[col] = pd.to_numeric(use[col], errors="coerce")
            use[col] = use[col].fillna(use[col].median())
    return use


def _fit_clustered(formula: str, data: pd.DataFrame):
    groups = pd.factorize(data["student"])[0]
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": groups})


def fit_parsons_models(df: pd.DataFrame) -> pd.DataFrame:
    """Parsons first-try quality vs exam %, three nested specifications."""
    use = _prep_model_frame(df)
    rows = []

    m1 = _fit_clustered("score_pct ~ parsons_first + C(cohort)", use)
    rows.append(
        _ols_row(m1, "cohort_fe", "parsons_first", len(use), use["student"].nunique(),
                 "Exam points per 1-point higher Parsons first-try %; semester x exam fixed effects; SEs clustered by student.")
    )

    controls = [c for c in ["active_days", "unique_chapters", "share_last_7d"] if c in use]
    rhs = ["parsons_first", "np.log1p(total_events)"] + controls
    m2 = _fit_clustered("score_pct ~ " + " + ".join(rhs) + " + C(cohort)", use)
    rows.append(
        _ols_row(m2, "cohort_fe_plus_effort", "parsons_first", len(use), use["student"].nunique(),
                 "Adds log events, active days, chapter coverage and last-week share as effort controls.")
    )

    # Prior ability: for midterm 2, control for the same student's midterm 1 score.
    mid1 = use.loc[use["midterm"].eq("mid1"), ["student", "score_pct"]].rename(columns={"score_pct": "mid1_score"})
    mid2 = use.loc[use["midterm"].eq("mid2")].merge(mid1, on="student", how="inner")
    if len(mid2) > 50:
        m3 = _fit_clustered("score_pct ~ parsons_first + mid1_score + np.log1p(total_events) + C(cohort)", mid2)
        rows.append(
            _ols_row(m3, "mid2_controls_for_mid1", "parsons_first", len(mid2), mid2["student"].nunique(),
                     "Midterm 2 only, controlling for the student's own midterm 1 score (prior ability) and volume.")
        )
    return pd.DataFrame(rows)


def quartile_table(df: pd.DataFrame, feature: str, target: str = "score_pct") -> pd.DataFrame:
    use = add_cohort(df)[["cohort", "semester_raw", "midterm", feature, target]].dropna().copy()
    use["feature_rank"] = use.groupby("cohort")[feature].rank(pct=True, method="average")
    use["feature_quartile"] = pd.cut(
        use["feature_rank"], bins=[0, 0.25, 0.5, 0.75, 1.0], labels=["Q1 bottom", "Q2", "Q3", "Q4 top"], include_lowest=True
    )
    summary = (
        use.groupby(["semester_raw", "midterm", "feature_quartile"], observed=False)
        .agg(n=(target, "size"), mean_score_pct=(target, "mean"))
        .reset_index()
    )
    summary["feature"] = feature
    return summary
