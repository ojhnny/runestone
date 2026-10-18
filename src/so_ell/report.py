"""Figures, headline metrics, dashboard tables and Looker Studio exports.

Everything here reads the files the earlier stages wrote, so it can be
re-run in seconds after tweaking a chart. Charts use one fixed palette:
blue = midterm 1, orange = midterm 2, a single blue ramp for magnitude,
gray for reference marks.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from so_ell import config, practice_exams, stats
from so_ell.scores import build_windows

T, F, P = config.TABLES_DIR, config.FIGURES_DIR, config.PROCESSED_DIR

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#0b0b0b", "#52514e"
MIDTERM_COLORS = {"mid1": BLUE, "mid2": ORANGE}
SEMESTER_ORDER = ["F21", "W22", "F22", "W23", "F23", "W24", "F24", "F25"]
FAMILY_LABELS = {
    "parsons": "Parsons",
    "activecode": "ActiveCode",
    "mchoice": "Multiple choice",
    "fillb": "Fill in blank",
    "clickable": "Clickable",
    "conceptcheck": "Concept checks",
    "dragndrop": "Drag and drop",
    "all_activity": "All activity",
    "learning": "All scored practice",
    "coverage": "Coverage",
    "consistency": "Consistency",
    "timing": "Timing",
    "mix": "Activity mix",
}


def label(feature: str) -> str:
    family, _, metric = feature.partition("__")
    return f"{FAMILY_LABELS.get(family, family)}: {metric.replace('_', ' ')}"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 170,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def save(fig, name: str) -> None:
    F.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(F / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cohort_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["semester_raw"] = pd.Categorical(df["semester_raw"], SEMESTER_ORDER, ordered=True)
    return df.sort_values(["semester_raw", "midterm"])


# ------------------------------------------------------------------ figures
def fig_pipeline_funnel(funnel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(funnel))[::-1]
    ax.barh(y, np.log10(funnel["count"]), color=BLUE, height=0.55)
    for yi, (_, row) in zip(y, funnel.iterrows(), strict=True):
        ax.text(np.log10(row["count"]) + 0.08, yi, f"{row['count']:,}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, funnel["label"])
    ax.set_xlabel("rows (log scale)")
    ax.set_xticks([])
    ax.set_xlim(0, np.log10(funnel["count"].max()) + 1.3)
    ax.grid(False)
    ax.set_title("From 19.8M log events to one row per student and exam")
    save(fig, "01_pipeline_funnel.png")


def fig_cohort_scores(summary: pd.DataFrame) -> None:
    df = _cohort_order(summary)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(SEMESTER_ORDER))
    for i, mid in enumerate(["mid1", "mid2"]):
        part = df.loc[df["midterm"].eq(mid)].set_index("semester_raw").reindex(SEMESTER_ORDER)
        ax.bar(x + (i - 0.5) * 0.38, part["mean_score_pct"], width=0.36, color=MIDTERM_COLORS[mid],
               label=f"Midterm {mid[-1]}")
    ax.set_xticks(x, SEMESTER_ORDER)
    ax.set_ylim(40, 90)
    ax.set_ylabel("mean reconstructed score (%)")
    ax.legend(ncol=2, loc="upper left")
    ax.set_title("Exam difficulty varies by semester, so every comparison is within a cohort")
    save(fig, "02_cohort_scores.png")


def fig_score_distributions(scores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, 100, 41)
    for mid in ["mid1", "mid2"]:
        ax.hist(scores.loc[scores["midterm"].eq(mid), "score_pct"], bins=bins, color=MIDTERM_COLORS[mid],
                alpha=0.55, label=f"Midterm {mid[-1]}", edgecolor="white", linewidth=0.6)
    ax.set_xlabel("reconstructed score (%)")
    ax.set_ylabel("student-exams")
    ax.legend()
    ax.set_title("Reconstructed midterm scores")
    save(fig, "03_score_distributions.png")


def fig_feature_signals(corr: pd.DataFrame) -> None:
    use = corr.loc[corr["q_value_bh"] < 0.05].copy()
    use = pd.concat([use.head(12), use.tail(6)]).drop_duplicates("feature").sort_values("within_group_rank_corr")
    fig, ax = plt.subplots(figsize=(9.5, 7))
    colors = np.where(use["within_group_rank_corr"] > 0, BLUE, ORANGE)
    ax.hlines(range(len(use)), 0, use["within_group_rank_corr"], color=colors, linewidth=2)
    ax.scatter(use["within_group_rank_corr"], range(len(use)), color=colors, s=48, zorder=3)
    ax.set_yticks(range(len(use)), [label(f) for f in use["feature"]])
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_xlabel("within-cohort rank correlation with midterm score")
    ax.set_title("Practice quality beats practice volume\n(features significant after false-discovery control)")
    save(fig, "04_feature_signals.png")


def fig_parsons_scatter(features: pd.DataFrame) -> None:
    use = stats.add_cohort(features).dropna(subset=[stats.PARSONS, "score_pct"]).copy()
    use["x"] = use.groupby("cohort")[stats.PARSONS].rank(pct=True) * 100
    use["y"] = use.groupby("cohort")["score_pct"].rank(pct=True) * 100
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.scatter(use["x"], use["y"], s=10, color=BLUE, alpha=0.25, linewidths=0)
    bins = pd.cut(use["x"], np.linspace(0, 100, 11), include_lowest=True)
    means = use.groupby(bins, observed=True)[["x", "y"]].mean()
    ax.plot(means["x"], means["y"], color=INK, linewidth=2, marker="o", markersize=6, label="decile mean")
    ax.set_xlabel("Parsons first-try success (within-cohort percentile)")
    ax.set_ylabel("midterm score (within-cohort percentile)")
    ax.legend(loc="upper left")
    ax.set_title("Students who solve Parsons problems on the first try\nscore higher on the midterm")
    save(fig, "05_parsons_scatter.png")


def fig_parsons_quartiles(quartiles: pd.DataFrame) -> None:
    use = quartiles.loc[quartiles["feature"].eq(stats.PARSONS)]
    agg = use.groupby("feature_quartile", observed=True).apply(
        lambda g: np.average(g["mean_score_pct"], weights=g["n"]), include_groups=False
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ramp = ["#b9d3f2", "#7fb0e8", "#4c8fdc", BLUE]
    ax.bar(agg.index.astype(str), agg.values, color=ramp, width=0.6)
    for i, v in enumerate(agg.values):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", color=INK)
    ax.set_ylim(40, 95)
    ax.set_ylabel("mean midterm score (%)")
    ax.set_xlabel("Parsons first-try quartile (cut within each cohort)")
    ax.set_title("Midterm score by Parsons first-try quartile")
    save(fig, "06_parsons_quartiles.png")


def fig_participation(features: pd.DataFrame) -> pd.DataFrame:
    families = ["conceptcheck", "mchoice", "fillb", "dragndrop", "activecode", "parsons"]
    rows = []
    for fam in families:
        col = f"{fam}__unique_problems"
        for mid, g in features.groupby("midterm"):
            rows.append(
                {
                    "family": FAMILY_LABELS[fam],
                    "midterm": mid,
                    "pct_used": 100 * g[col].fillna(0).gt(0).mean(),
                    "median_unique_problems": g[col].fillna(0).median(),
                }
            )
    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(families))
    for i, mid in enumerate(["mid1", "mid2"]):
        part = summary.loc[summary["midterm"].eq(mid)]
        axes[0].bar(x + (i - 0.5) * 0.38, part["pct_used"], width=0.36, color=MIDTERM_COLORS[mid], label=f"Midterm {mid[-1]}")
        axes[1].bar(x + (i - 0.5) * 0.38, part["median_unique_problems"], width=0.36, color=MIDTERM_COLORS[mid])
    for ax in axes:
        ax.set_xticks(x, [FAMILY_LABELS[f] for f in families], rotation=20)
    axes[0].set_ylim(80, 100.5)
    axes[0].set_ylabel("% of students with any pre-exam use")
    axes[0].set_title("Almost everyone used every activity type")
    axes[1].set_ylabel("median distinct problems before the exam")
    axes[1].set_title("…so volume of use, not use itself, varies")
    axes[0].legend(loc="lower left")
    save(fig, "07_participation_and_exposure.png")
    return summary


def fig_effort_efficiency(features: pd.DataFrame) -> pd.DataFrame:
    use = stats.add_cohort(features).dropna(subset=["parsons__total_events", stats.PARSONS, "score_pct"]).copy()
    use["effort"] = use.groupby("cohort")["parsons__total_events"].rank(pct=True) * 100
    use["efficiency"] = use.groupby("cohort")[stats.PARSONS].rank(pct=True) * 100
    use["profile"] = np.select(
        [(use["effort"] > 50) & (use["efficiency"] > 50), (use["effort"] <= 50) & (use["efficiency"] > 50),
         (use["effort"] > 50) & (use["efficiency"] <= 50)],
        ["High effort, high efficiency", "Low effort, high efficiency", "High effort, low efficiency"],
        default="Low effort, low efficiency",
    )
    summary = (
        use.groupby("profile", as_index=False)
        .agg(n_students=("anon_student_id", "count"), mean_exam_score=("score_pct", "mean"), median_exam_score=("score_pct", "median"))
        .sort_values("mean_exam_score")
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    colors = [ORANGE if "low efficiency" in p else BLUE for p in summary["profile"]]
    ax.barh(summary["profile"], summary["mean_exam_score"], color=colors, height=0.55)
    for i, v in enumerate(summary["mean_exam_score"]):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", color=INK)
    ax.set_xlim(40, 90)
    ax.set_xlabel("mean midterm score (%)")
    ax.set_title("Getting Parsons right matters more than doing a lot of them")
    save(fig, "08_effort_vs_efficiency.png")
    return summary


def fig_parsons_baseline(features: pd.DataFrame) -> pd.DataFrame:
    use = stats.add_cohort(features).dropna(subset=[stats.PARSONS, "parsons__mean_score_gain", "score_pct"]).copy()
    use["quartile"] = use.groupby("cohort")[stats.PARSONS].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    )
    summary = use.groupby("quartile", observed=True, as_index=False).agg(
        n_students=("anon_student_id", "count"),
        mean_parsons_first=(stats.PARSONS, "mean"),
        mean_parsons_gain=("parsons__mean_score_gain", "mean"),
        mean_exam_score=("score_pct", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(summary["quartile"].astype(str), summary["mean_parsons_gain"], color=AQUA, width=0.6)
    axes[0].set_title("Students who started lower improved more…")
    axes[0].set_ylabel("mean Parsons score gain (first → last)")
    axes[1].bar(summary["quartile"].astype(str), summary["mean_exam_score"], color=BLUE, width=0.6)
    axes[1].set_ylim(40, 95)
    axes[1].set_title("…but still scored lower on the midterm")
    axes[1].set_ylabel("mean midterm score (%)")
    for ax in axes:
        ax.set_xlabel("Parsons first-try quartile")
    save(fig, "09_parsons_gain_paradox.png")
    return summary


def fig_cramming(bins: pd.DataFrame) -> None:
    use = bins.loc[bins["score_group"].isin(["Q1 lowest", "Q4 highest"])]
    order = ["15-28 days before", "8-14 days before", "4-7 days before", "1-3 days before"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(order))
    for i, (grp, color) in enumerate([("Q1 lowest", ORANGE), ("Q4 highest", BLUE)]):
        part = use.loc[use["score_group"].eq(grp)].set_index("timing_bin").reindex(order)
        ax.bar(x + (i - 0.5) * 0.38, part["mean_share_pct"], width=0.36, color=color, label=f"{grp} exam quartile")
    ax.set_xticks(x, order)
    ax.set_ylabel("% of last-28-day practice")
    ax.legend()
    ax.set_title("Low and high scorers time their practice about the same way")
    save(fig, "10_cramming_vs_spacing.png")


def fig_practice_exams(group: pd.DataFrame) -> None:
    order = ["No timed practice", "Started only", "Finished timed practice"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(order))
    for i, mid in enumerate(["mid1", "mid2"]):
        part = group.loc[group["midterm"].eq(mid)].set_index("practice_group").reindex(order)
        ax.bar(x + (i - 0.5) * 0.38, part["mean_exam_score"], width=0.36, color=MIDTERM_COLORS[mid], label=f"Midterm {mid[-1]}")
    ax.set_xticks(x, order)
    ax.set_ylim(40, 90)
    ax.set_ylabel("mean midterm score (%)")
    ax.legend(ncol=2, loc="upper left")
    ax.set_title("Students who finished a timed practice exam scored higher")
    save(fig, "11_practice_exams.png")


def fig_scoring_coverage(unscored: pd.DataFrame, scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(scores["coverage_pct"], bins=np.linspace(50, 100, 26), color=BLUE, edgecolor="white")
    axes[0].set_xlabel("% of a student's exam items that could be scored")
    axes[0].set_ylabel("student-exams")
    axes[0].set_title("Most exams are almost fully scoreable")
    top = unscored.head(8).iloc[::-1]
    axes[1].barh(top["score_source"].str.replace("_", " "), top["student_problem_rows"], color=GRAY)
    axes[1].set_xlabel("student-item rows left unscored")
    axes[1].set_title("Why items go unscored")
    save(fig, "12_scoring_coverage.png")


def fig_models(comparison: pd.DataFrame, lead: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    names = {
        "logistic": "Logistic, practice features",
        "gradient_boosting": "Gradient boosting, practice features",
        "mid2_mid1_only": "Mid 2: mid 1 score only",
        "mid2_practice_only": "Mid 2: practice only",
        "mid2_mid1_plus_practice": "Mid 2: mid 1 + practice",
    }
    comp = comparison.set_index("model").reindex(list(names)).iloc[::-1]
    axes[0].barh([names[m] for m in comp.index], comp["auc"], color=[BLUE, BLUE, GRAY, BLUE, BLUE][::-1], height=0.55)
    for i, v in enumerate(comp["auc"]):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", color=INK)
    axes[0].set_xlim(0.5, 0.95)
    axes[0].axvline(0.5, color=GRAY, linewidth=1)
    axes[0].set_xlabel("leave-one-semester-out ROC AUC")
    axes[0].set_title("Flagging bottom-quartile students")
    for kind, color in [("logistic", BLUE), ("gradient_boosting", ORANGE)]:
        part = lead.loc[lead["model"].eq(kind)].sort_values("cutoff_days", ascending=False)
        axes[1].plot(part["cutoff_days"], part["auc"], marker="o", color=color, linewidth=2,
                     label="Logistic" if kind == "logistic" else "Gradient boosting")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("days before the exam the data stops")
    axes[1].set_ylabel("ROC AUC")
    axes[1].legend()
    axes[1].set_title("How early can we flag them?")
    save(fig, "13_at_risk_models.png")


def fig_calibration(cal: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    for kind, color in [("logistic", BLUE), ("gradient_boosting", ORANGE)]:
        part = cal.loc[cal["model"].eq(kind)]
        ax.plot(part["mean_predicted"], part["observed_rate"], marker="o", color=color, linewidth=2,
                label="Logistic" if kind == "logistic" else "Gradient boosting")
    ax.set_xlabel("predicted risk (decile mean)")
    ax.set_ylabel("observed bottom-quartile rate")
    ax.legend()
    ax.set_title("Calibration, out of fold")
    save(fig, "14_calibration.png")


def fig_sensitivity(sens: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(sens))[::-1]
    ax.errorbar(sens["fe_coef"] * 10, y, xerr=[(sens["fe_coef"] - sens["fe_ci_low"]) * 10, (sens["fe_ci_high"] - sens["fe_coef"]) * 10],
                fmt="o", color=BLUE, ecolor=BLUE, capsize=4, markersize=7)
    ax.set_yticks(y, sens["label"])
    ax.set_xlabel("exam points per +10 points of Parsons first-try success (95% CI)")
    ax.set_title("The Parsons result does not depend on the answer-key threshold")
    save(fig, "15_key_threshold_sensitivity.png")


# ------------------------------------------------------------------ tables
def pipeline_funnel() -> pd.DataFrame:
    counts = pd.read_csv(config.QUALITY_DIR / "ingest_row_counts.csv")
    attempts = pd.read_parquet(config.EXAM_ATTEMPTS_PARQUET, columns=["session_id"])
    problems = pd.read_parquet(P / "student_exam_problem_scores.parquet", columns=["problem_name"])
    scores = pd.read_parquet(P / "student_exam_scores.parquet", columns=["anon_student_id"])
    students = pd.read_parquet(P / "student_exam_scores.parquet", columns=["semester_raw", "anon_student_id"]).drop_duplicates()
    rows = [
        ("Raw Runestone log events", int(counts["rows"].sum()), "Every row in the 13 Datashop exports."),
        ("Timed real-midterm attempts", len(attempts), "Start/finish markers for the real (not practice) timed midterm."),
        ("Scored student-exam items", len(problems), "One row per student, exam and problem answered during the exam."),
        ("Student-exam scores", len(scores), "One reconstructed score per student and midterm."),
        ("Distinct students", len(students), "Students with at least one scored midterm, 8 semesters."),
    ]
    return pd.DataFrame(rows, columns=["label", "count", "description"])


def headline_metrics(corr, fe, boot, comparison, lead, practice_signals, funnel) -> pd.DataFrame:
    parsons = corr.loc[corr["feature"].eq(stats.PARSONS)].iloc[0]
    spec = fe.set_index("spec")
    lead_gbm = lead.loc[lead["model"].eq("logistic")].set_index("cutoff_days")["auc"]
    comp = comparison.set_index("model")
    rows = [
        ("raw_events", funnel.iloc[0]["count"], "Rows across all raw exports."),
        ("student_exams", funnel.iloc[3]["count"], "Reconstructed student-midterm scores."),
        ("students", funnel.iloc[4]["count"], "Distinct students."),
        ("parsons_quartile_gap", boot["estimate"], "Top vs bottom quartile of Parsons first-try success, within cohort (exam points)."),
        ("parsons_quartile_gap_ci_low", boot["ci_low"], "95% cluster-bootstrap CI, lower."),
        ("parsons_quartile_gap_ci_high", boot["ci_high"], "95% cluster-bootstrap CI, upper."),
        ("parsons_rank_corr", parsons["within_group_rank_corr"], "Within-cohort rank correlation with midterm score."),
        ("parsons_fe_coef_per_10pts", 10 * spec.loc["cohort_fe", "coef"], "Exam points per +10 Parsons points, cohort FE, clustered SEs."),
        ("parsons_controlled_coef_per_10pts", 10 * spec.loc["cohort_fe_plus_effort", "coef"], "Same, plus effort controls."),
        ("parsons_mid1_controlled_coef_per_10pts", 10 * spec.loc["mid2_controls_for_mid1", "coef"], "Midterm 2, controlling for midterm 1 score."),
        ("at_risk_auc_logistic", comp.loc["logistic", "auc"], "Leave-one-semester-out AUC, logistic."),
        ("at_risk_auc_gbm", comp.loc["gradient_boosting", "auc"], "Leave-one-semester-out AUC, gradient boosting."),
        ("at_risk_precision_top10_logistic", comp.loc["logistic", "precision_top10pct"], "Share truly at risk among the 10% flagged."),
        ("at_risk_auc_14d_before", lead_gbm.get(14, np.nan), "Logistic AUC using only practice >= 14 days before the exam."),
        ("mid2_auc_mid1_only", comp.loc["mid2_mid1_only", "auc"], "Midterm 2 risk from midterm 1 score alone."),
        ("mid2_auc_mid1_plus_practice", comp.loc["mid2_mid1_plus_practice", "auc"], "Midterm 2 risk, mid1 score + practice."),
        ("practice_exam_finish_gap", practice_signals.set_index("metric").loc["finished_vs_none_gap", "value"],
         "Finished a timed practice exam vs none, within-cohort gap (points)."),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


def export_looker(features, corr, fe, comparison, chapter, timing, quartiles, summary) -> None:
    L = config.LOOKER_DIR
    L.mkdir(parents=True, exist_ok=True)
    summary.assign(cohort=summary["semester_raw"] + "_" + summary["midterm"]).to_csv(L / "cohort_kpis.csv", index=False)
    corr.to_csv(L / "feature_score_signals.csv", index=False)
    fe.to_csv(L / "parsons_fixed_effects.csv", index=False)
    comparison.to_csv(L / "at_risk_model_comparison.csv", index=False)
    chapter.to_csv(L / "chapter_activity_by_cohort.csv", index=False)
    timing.to_csv(L / "practice_timing_by_score_band.csv", index=False)
    quartiles.to_csv(L / "feature_quartile_exam_scores.csv", index=False)
    for name in ["events_by_semester.csv", "event_log_overview.csv"]:
        pd.read_csv(T / name).to_csv(L / name, index=False)


def build_all() -> None:
    style()
    scores = pd.read_parquet(P / "student_exam_scores.parquet")
    features = pd.read_parquet(P / "student_exam_features.parquet")
    windows = build_windows(pd.read_parquet(config.EXAM_ATTEMPTS_PARQUET))
    corr = pd.read_csv(T / "feature_signals.csv")
    fe = pd.read_csv(T / "parsons_models.csv")
    boot = pd.read_csv(T / "parsons_gap_bootstrap.csv").iloc[0]
    quartiles = pd.read_csv(T / "feature_quartile_scores.csv")
    summary = pd.read_csv(T / "cohort_score_summary.csv")
    comparison = pd.read_csv(T / "at_risk_model_comparison.csv")
    lead = pd.read_csv(T / "at_risk_lead_time.csv")
    cal = pd.read_csv(T / "at_risk_calibration.csv")
    sens = pd.read_csv(T / "key_threshold_sensitivity.csv")
    unscored = pd.read_csv(T / "unscored_items_summary.csv")

    practice = practice_exams.practice_midterm_analysis(scores)
    practice["student"].to_parquet(P / "practice_exam_students.parquet", index=False)
    practice["group"].to_csv(T / "practice_exam_groups.csv", index=False)
    practice["cohort"].to_csv(T / "practice_exam_by_cohort.csv", index=False)
    practice["signals"].to_csv(T / "practice_exam_signals.csv", index=False)
    cram = practice_exams.cramming_analysis(scores, windows)
    cram["metrics"].to_parquet(P / "cramming_student_metrics.parquet", index=False)
    cram["bins"].to_csv(T / "cramming_timing_bins.csv", index=False)
    cram["summary"].to_csv(T / "cramming_summary.csv", index=False)

    funnel = pipeline_funnel()
    funnel.to_csv(T / "pipeline_funnel.csv", index=False)

    fig_pipeline_funnel(funnel)
    fig_cohort_scores(summary)
    fig_score_distributions(scores)
    fig_feature_signals(corr)
    fig_parsons_scatter(features)
    fig_parsons_quartiles(quartiles)
    fig_participation(features).to_csv(T / "participation_and_exposure.csv", index=False)
    fig_effort_efficiency(features).to_csv(T / "parsons_effort_profiles.csv", index=False)
    fig_parsons_baseline(features).to_csv(T / "parsons_gain_by_quartile.csv", index=False)
    fig_cramming(cram["bins"])
    fig_practice_exams(practice["group"])
    fig_scoring_coverage(unscored, scores)
    fig_models(comparison, lead)
    fig_calibration(cal)
    fig_sensitivity(sens)

    headline = headline_metrics(corr, fe, boot, comparison, lead, practice["signals"], funnel)
    headline.to_csv(config.OUTPUTS_DIR / "headline_metrics.csv", index=False)
    export_looker(features, corr, fe, comparison, pd.read_csv(T / "chapter_activity_by_cohort.csv"),
                  pd.read_csv(T / "practice_timing_by_score_band.csv"), quartiles, summary)

