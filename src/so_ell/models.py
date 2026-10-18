"""At-risk prediction: can pre-exam ebook activity flag students headed for
the bottom quartile of their exam?

Design choices:
- Target is "bottom quartile *within the cohort*", so an easy semester and a
  hard semester each contribute their own weakest 25%.
- Validation is leave-one-semester-out. Each fold trains on 7 semesters and
  predicts a semester the model has never seen, which is how an instructor
  would use it (train on past terms, score the current one).
- Two model families: a regularized logistic regression (interpretable
  baseline) and a histogram gradient-boosted tree (captures nonlinearity and
  missing values natively). Metrics: ROC AUC, average precision, Brier score,
  and precision among the 10% of students flagged as highest risk.
- Every run is logged to MLflow when it is installed (``uv sync --extra tracking``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from so_ell import config
from so_ell.stats import add_cohort

PRACTICE_FEATURES = [
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


def make_model(kind: str) -> Pipeline:
    if kind == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0)),
            ]
        )
    if kind == "gradient_boosting":
        return Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                        early_stopping=False, random_state=config.RANDOM_SEED,
                    ),
                )
            ]
        )
    raise ValueError(kind)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = add_cohort(df)
    out["score_rank"] = out.groupby("cohort")["score_pct"].rank(pct=True)
    out["at_risk"] = (out["score_rank"] <= 0.25).astype(int)
    return out


def precision_at_top(y: np.ndarray, proba: np.ndarray, share: float = 0.10) -> float:
    k = max(1, int(round(len(y) * share)))
    top = np.argsort(-proba)[:k]
    return float(y[top].mean())


@dataclass
class CVResult:
    name: str
    folds: pd.DataFrame
    summary: dict[str, float]
    oof: pd.DataFrame
    importance: pd.DataFrame | None


def leave_one_semester_out(df: pd.DataFrame, features: list[str], kind: str, name: str) -> CVResult:
    use = add_target(df).dropna(subset=["at_risk"]).reset_index(drop=True)
    features = [f for f in features if f in use.columns]
    X, y = use[features], use["at_risk"].to_numpy()
    groups = use["semester_raw"].astype(str).to_numpy()
    oof = np.full(len(use), np.nan)
    rows = []
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        model = make_model(kind).fit(X.iloc[train], y[train])
        proba = model.predict_proba(X.iloc[test])[:, 1]
        oof[test] = proba
        rows.append(
            {
                "model": name,
                "held_out_semester": groups[test][0],
                "n_test": len(test),
                "auc": roc_auc_score(y[test], proba),
                "average_precision": average_precision_score(y[test], proba),
            }
        )
    summary = {
        "auc": roc_auc_score(y, oof),
        "average_precision": average_precision_score(y, oof),
        "brier": brier_score_loss(y, oof),
        "precision_top10pct": precision_at_top(y, oof),
        "base_rate": float(y.mean()),
        "n": len(use),
        "n_features": len(features),
    }
    importance = None
    if kind == "gradient_boosting":
        full = make_model(kind).fit(X, y)
        imp = permutation_importance(full, X, y, scoring="roc_auc", n_repeats=10, random_state=config.RANDOM_SEED)
        importance = pd.DataFrame(
            {"feature": features, "auc_drop_mean": imp.importances_mean, "auc_drop_sd": imp.importances_std}
        ).sort_values("auc_drop_mean", ascending=False)
    elif kind == "logistic":
        full = make_model(kind).fit(X, y)
        importance = pd.DataFrame({"feature": features, "std_coef": full.named_steps["clf"].coef_[0]}).sort_values("std_coef")
    oof_frame = use[["semester_raw", "anon_student_id", "midterm", "at_risk"]].assign(model=name, proba=oof)
    return CVResult(name, pd.DataFrame(rows), summary, oof_frame, importance)


def calibration_table(oof: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    use = oof.copy()
    use["bin"] = pd.qcut(use["proba"].rank(method="first"), bins, labels=False) + 1
    return (
        use.groupby(["model", "bin"], as_index=False)
        .agg(mean_predicted=("proba", "mean"), observed_rate=("at_risk", "mean"), n=("at_risk", "size"))
    )


def run_model_suite(features: pd.DataFrame) -> dict[str, CVResult]:
    """Main comparison: logistic vs gradient boosting, plus a prior-ability baseline."""
    results = {
        "logistic": leave_one_semester_out(features, PRACTICE_FEATURES, "logistic", "logistic"),
        "gradient_boosting": leave_one_semester_out(features, PRACTICE_FEATURES, "gradient_boosting", "gradient_boosting"),
    }

    # Midterm 2 only: does ebook practice add anything once we know the mid1 score?
    mid = add_cohort(features)
    mid1 = mid.loc[mid["midterm"].eq("mid1"), ["semester_raw", "anon_student_id", "score_pct"]].rename(
        columns={"score_pct": "mid1_score"}
    )
    mid2 = mid.loc[mid["midterm"].eq("mid2")].merge(mid1, on=["semester_raw", "anon_student_id"], how="inner")
    results["mid2_mid1_only"] = leave_one_semester_out(mid2, ["mid1_score"], "logistic", "mid2_mid1_only")
    results["mid2_practice_only"] = leave_one_semester_out(mid2, PRACTICE_FEATURES, "logistic", "mid2_practice_only")
    results["mid2_mid1_plus_practice"] = leave_one_semester_out(
        mid2, ["mid1_score"] + PRACTICE_FEATURES, "logistic", "mid2_mid1_plus_practice"
    )
    return results


def lead_time_curve(features_by_cutoff: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Out-of-fold AUC when the model only sees practice from >= N days before the exam."""
    rows = []
    for cutoff, frame in sorted(features_by_cutoff.items()):
        for kind in ["logistic", "gradient_boosting"]:
            result = leave_one_semester_out(frame, PRACTICE_FEATURES, kind, f"{kind}_cutoff_{cutoff}")
            rows.append({"cutoff_days": cutoff, "model": kind} | result.summary)
    return pd.DataFrame(rows)


def log_to_mlflow(results: dict[str, CVResult], lead_time: pd.DataFrame | None = None) -> bool:
    """Log each model as an MLflow run. Returns False when MLflow isn't installed."""
    try:
        import mlflow
    except ImportError:
        return False
    config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(config.MLRUNS_DIR / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("so-ell-at-risk")
    for name, result in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "validation": "leave_one_semester_out", "n_features": result.summary["n_features"]})
            mlflow.log_metrics({k: float(v) for k, v in result.summary.items() if k not in {"n", "n_features"}})
            for row in result.folds.itertuples():
                mlflow.log_metric(f"auc_{row.held_out_semester}", float(row.auc))
            if result.importance is not None:
                mlflow.log_text(result.importance.to_csv(index=False), "importance.csv")
    if lead_time is not None:
        with mlflow.start_run(run_name="lead_time_curve"):
            for row in lead_time.itertuples():
                mlflow.log_metric(f"auc_{row.model}", float(row.auc), step=int(row.cutoff_days))
            mlflow.log_text(lead_time.to_csv(index=False), "lead_time.csv")
    return True
