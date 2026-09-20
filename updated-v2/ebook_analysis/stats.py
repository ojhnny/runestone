"""Cohort stats and a small at-risk model.

The original 17.5-point Parsons gap is a within-cohort quartile comparison.
That's easy to explain to instructors, but interviewers will ask whether it
just picks up harder semesters or students who practice more. These specs
are the follow-up:

1. OLS with semester x exam (cohort) fixed effects
2. Same thing plus volume / consistency controls
3. A leave-one-semester-out logistic model for "bottom quartile on the exam"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def add_cohort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cohort"] = out["semester_raw"].astype(str) + "_" + out["midterm"].astype(str)
    return out


def within_group_rank_corr(df: pd.DataFrame, group_col: str, feature: str, target: str) -> float:
    use = df[[group_col, feature, target]].dropna().copy()
    if len(use) < 10 or use[feature].nunique() < 3 or use[target].nunique() < 3:
        return np.nan
    use["x_rank"] = use.groupby(group_col)[feature].rank(pct=True)
    use["y_rank"] = use.groupby(group_col)[target].rank(pct=True)
    return float(use[["x_rank", "y_rank"]].corr().iloc[0, 1])


def within_group_quartile_gap(df: pd.DataFrame, group_col: str, feature: str, target: str) -> float:
    use = df[[group_col, feature, target]].dropna().copy()
    if len(use) < 20 or use[feature].nunique() < 4:
        return np.nan
    use["feature_rank"] = use.groupby(group_col)[feature].rank(pct=True, method="average")
    bottom = use.loc[use["feature_rank"] <= 0.25, target]
    top = use.loc[use["feature_rank"] > 0.75, target]
    if bottom.empty or top.empty:
        return np.nan
    return float(top.mean() - bottom.mean())


def compute_corr_table(df: pd.DataFrame, group_col: str, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        use = df[[group_col, feature, target]].dropna()
        if use.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "n": int(len(use)),
                "within_group_rank_corr": within_group_rank_corr(df, group_col, feature, target),
                "top_quartile_minus_bottom_quartile_pct_points": within_group_quartile_gap(
                    df, group_col, feature, target
                ),
            }
        )
    out = pd.DataFrame(rows).dropna(subset=["within_group_rank_corr"])
    return out.sort_values("within_group_rank_corr", ascending=False).reset_index(drop=True)


def _ols_row(model, spec: str, term: str, n: int, note: str) -> dict[str, object]:
    ci = model.conf_int().loc[term]
    return {
        "spec": spec,
        "term": term,
        "n": n,
        "coef": float(model.params[term]),
        "std_err": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "r_squared": float(model.rsquared),
        "note": note,
    }


def fit_parsons_models(df: pd.DataFrame) -> pd.DataFrame:
    """Parsons first-try quality, with and without effort controls."""
    use = add_cohort(df)
    # statsmodels formulas hate the double underscores, so copy to plain names.
    use = use.rename(
        columns={
            "parsons__mean_first_score": "parsons_first",
            "all_activity__total_events": "total_events",
            "all_activity__active_days": "active_days",
            "coverage__unique_chapters": "unique_chapters",
            "timing__share_last_7d": "share_last_7d",
        }
    )
    needed = ["score_pct", "parsons_first", "cohort"]
    use = use.dropna(subset=needed).copy()
    # 0-1 score -> percentage points, so the coefficient reads like exam points.
    if use["parsons_first"].max() <= 1.5:
        use["parsons_first"] = use["parsons_first"] * 100

    rows = []
    m1 = smf.ols("score_pct ~ parsons_first + C(cohort)", data=use).fit()
    rows.append(
        _ols_row(
            m1,
            "cohort_fe",
            "parsons_first",
            len(use),
            "Exam points associated with a 1-point increase in Parsons first-try %; semester x exam FE.",
        )
    )

    extra = ["total_events", "active_days", "unique_chapters", "share_last_7d"]
    extra = [col for col in extra if col in use.columns]
    controls = use.dropna(subset=["parsons_first", "score_pct"] + [c for c in extra if c in {"total_events", "active_days"}]).copy()
    for col in extra:
        controls[col] = pd.to_numeric(controls[col], errors="coerce")
        controls[col] = controls[col].fillna(controls[col].median())

    rhs = ["parsons_first"]
    if "total_events" in extra:
        rhs.append("np.log1p(total_events)")
    rhs.extend(col for col in extra if col != "total_events")
    formula = "score_pct ~ " + " + ".join(rhs) + " + C(cohort)"
    m2 = smf.ols(formula, data=controls).fit()
    rows.append(
        _ols_row(
            m2,
            "cohort_fe_plus_effort",
            "parsons_first",
            len(controls),
            "Same FE model, also controlling for log events, active days, chapter coverage, and last-week share.",
        )
    )
    return pd.DataFrame(rows)


def quartile_table(df: pd.DataFrame, feature: str, target: str = "score_pct") -> pd.DataFrame:
    use = add_cohort(df)[["cohort", "semester_raw", "midterm", feature, target]].dropna().copy()
    use["feature_rank"] = use.groupby("cohort")[feature].rank(pct=True, method="average")
    use["feature_quartile"] = pd.cut(
        use["feature_rank"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["Q1 bottom", "Q2", "Q3", "Q4 top"],
        include_lowest=True,
    )
    summary = (
        use.groupby(["semester_raw", "midterm", "feature_quartile"], observed=False)
        .agg(n=("score_pct", "size"), mean_score_pct=(target, "mean"))
        .reset_index()
    )
    summary["feature"] = feature
    return summary


def at_risk_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict bottom-quartile exam score, leaving one semester out each fold."""
    use = add_cohort(df)
    use["score_rank"] = use.groupby("cohort")["score_pct"].rank(pct=True)
    use["at_risk"] = (use["score_rank"] <= 0.25).astype(int)

    feature_cols = [
        col
        for col in [
            "parsons__mean_first_score",
            "mchoice__mean_first_score",
            "activecode__mean_best_score",
            "all_activity__total_events",
            "all_activity__active_days",
            "coverage__unique_chapters",
            "coverage__chapter_diversity",
            "timing__share_last_7d",
            "consistency__problems_per_active_day",
            "mix__coding_share",
        ]
        if col in use.columns
    ]
    model_df = use.dropna(subset=["at_risk", "semester_raw"]).copy()
    X = model_df[feature_cols]
    y = model_df["at_risk"].to_numpy()
    groups = model_df["semester_raw"].astype(str).to_numpy()

    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )

    n_groups = pd.Series(groups).nunique()
    splitter = GroupKFold(n_splits=min(5, n_groups))
    fold_rows = []
    oof = np.full(len(model_df), np.nan)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = proba
        fold_rows.append(
            {
                "fold": fold,
                "held_out_semesters": ", ".join(sorted(set(groups[test_idx]))),
                "n_test": int(len(test_idx)),
                "auc": float(roc_auc_score(y[test_idx], proba)),
            }
        )

    # Fit once on all data for a readable coefficient table. The AUC above
    # is the number to trust; these signs are just for interpretation.
    pipe.fit(X, y)
    coefs = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": pipe.named_steps["clf"].coef_[0],
        }
    ).sort_values("coef")
    coefs["meaning"] = np.where(
        coefs["coef"] > 0,
        "Higher value -> more likely bottom-quartile exam",
        "Higher value -> less likely bottom-quartile exam",
    )

    oof_auc = float(roc_auc_score(y, oof))
    metrics = pd.DataFrame(fold_rows)
    metrics.loc[len(metrics)] = {
        "fold": "overall_oof",
        "held_out_semesters": "all (out of fold)",
        "n_test": int(len(model_df)),
        "auc": oof_auc,
    }
    return metrics, coefs
