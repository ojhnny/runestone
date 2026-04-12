from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "analysis8_outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

MIDTERM_PALETTE = {"mid1": "#1b9e77", "mid2": "#d95f02"}
FAMILY_LABELS = {
    "parsons": "Parsons",
    "activecode": "ActiveCode",
    "mchoice": "Multiple Choice",
    "fillb": "Fill In Blank",
    "clickable": "Clickable",
    "conceptcheck": "Concept Checks",
    "dragndrop": "Drag and Drop",
    "all_activity": "All Activities",
    "learning": "All Scored Practice",
}
METRIC_LABELS = {
    "total_events": "Total events",
    "active_days": "Active days",
    "unique_problems": "Unique problems",
    "avg_event_score": "Average event score",
    "mean_first_score": "Mean first score",
    "mean_best_score": "Mean best score",
    "mean_score_gain": "Mean score gain",
    "mean_events_per_problem": "Events per problem",
}


def clean_problem_name(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def parse_midterm_from_name(name: object) -> str | None:
    cleaned = clean_problem_name(name)
    if re.search(r"\b(?:m|mid|midterm)\s*1\b", cleaned):
        return "mid1"
    if re.search(r"\b(?:m|mid|midterm)\s*2\b", cleaned):
        return "mid2"
    return None


def parse_mchoice_score(action: object) -> float:
    text = str(action)
    if ":correct" in text:
        return 1.0
    if ":no" in text:
        return 0.0
    return np.nan


def parse_parsons_score(action: object) -> float:
    text = str(action)
    if text.startswith("correct"):
        return 1.0
    if text.startswith("incorrect"):
        return 0.0
    return np.nan


def parse_hparsons_score(action: object) -> float:
    try:
        obj = json.loads(str(action))
    except Exception:
        obj = None
    if isinstance(obj, dict):
        if "percent" in obj:
            try:
                return float(obj["percent"])
            except Exception:
                return np.nan
        if "correct" in obj:
            return 1.0 if str(obj["correct"]).upper().startswith("T") else 0.0
    match = re.search(r'"percent"\s*:\s*([0-9.]+)', str(action))
    return float(match.group(1)) if match else np.nan


def parse_unittest_pct(action: object) -> float:
    match = re.search(r"percent:([0-9.]+)", str(action))
    return float(match.group(1)) / 100 if match else np.nan


def within_group_rank_corr_generic(
    df: pd.DataFrame, group_col: str, feature: str, target: str
) -> float:
    use = df[[group_col, feature, target]].dropna().copy()
    if len(use) < 10 or use[feature].nunique() < 3 or use[target].nunique() < 3:
        return np.nan
    use["x_rank"] = use.groupby(group_col)[feature].rank(pct=True)
    use["y_rank"] = use.groupby(group_col)[target].rank(pct=True)
    return use[["x_rank", "y_rank"]].corr().iloc[0, 1]


def within_group_quartile_gap_generic(
    df: pd.DataFrame, group_col: str, feature: str, target: str
) -> float:
    use = df[[group_col, feature, target]].dropna().copy()
    if len(use) < 20 or use[feature].nunique() < 4:
        return np.nan
    use["feature_rank"] = use.groupby(group_col)[feature].rank(pct=True, method="average")
    bottom = use.loc[use["feature_rank"] <= 0.25, target]
    top = use.loc[use["feature_rank"] > 0.75, target]
    if bottom.empty or top.empty:
        return np.nan
    return top.mean() - bottom.mean()


def semester_sort_key(code: str) -> tuple[int, int]:
    season = 0 if str(code).startswith("W") else 1
    year = 2000 + int(str(code)[1:])
    return (year, season)


def ordered_semesters(values: list[str]) -> list[str]:
    return sorted(pd.Series(values).dropna().unique().tolist(), key=semester_sort_key)


def clean_feature_label(feature: str) -> str:
    family, metric = feature.split("__", 1)
    return f"{FAMILY_LABELS.get(family, family.title())}: {METRIC_LABELS.get(metric, metric.replace('_', ' ').title())}"


def load_outputs() -> dict[str, pd.DataFrame]:
    return {
        "scores": pd.read_parquet(OUT_DIR / "student_midterm_scores.parquet"),
        "features": pd.read_parquet(OUT_DIR / "student_midterm_activity_features.parquet"),
        "semester_summary": pd.read_csv(OUT_DIR / "semester_midterm_score_summary.csv"),
        "activity_corr": pd.read_csv(OUT_DIR / "activity_correlations.csv"),
        "change_corr": pd.read_csv(OUT_DIR / "activity_change_correlations.csv"),
        "unsupported": pd.read_csv(OUT_DIR / "unsupported_or_ambiguous_exam_items.csv"),
    }


def prep_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    scores = frames["scores"].copy()
    features = frames["features"].copy()
    semester_summary = frames["semester_summary"].copy()
    activity_corr = frames["activity_corr"].copy()
    change_corr = frames["change_corr"].copy()
    unsupported = frames["unsupported"].copy()

    semester_order = ordered_semesters(scores["semester_raw"].tolist())
    scores["semester_raw"] = pd.Categorical(scores["semester_raw"], categories=semester_order, ordered=True)
    scores["midterm"] = pd.Categorical(scores["midterm"], categories=["mid1", "mid2"], ordered=True)
    scores["cohort_label"] = scores["semester_raw"].astype(str) + " " + scores["midterm"].astype(str).str.upper()
    scores["cohort"] = scores["semester_raw"].astype(str) + "_" + scores["midterm"].astype(str)

    features["semester_raw"] = pd.Categorical(features["semester_raw"], categories=semester_order, ordered=True)
    features["midterm"] = pd.Categorical(features["midterm"], categories=["mid1", "mid2"], ordered=True)
    features["cohort"] = features["semester_raw"].astype(str) + "_" + features["midterm"].astype(str)
    features["score_rank_pct"] = features.groupby("cohort")["score_pct"].rank(pct=True) * 100

    semester_summary["semester_raw"] = pd.Categorical(semester_summary["semester_raw"], categories=semester_order, ordered=True)
    semester_summary["midterm"] = pd.Categorical(semester_summary["midterm"], categories=["mid1", "mid2"], ordered=True)

    activity_corr["feature_label"] = activity_corr["feature"].map(clean_feature_label)
    change_corr["feature_label"] = change_corr["feature"].str.replace("__change", "", regex=False).map(clean_feature_label)

    return {
        "scores": scores.sort_values(["semester_raw", "midterm", "anon_student_id"]),
        "features": features.sort_values(["semester_raw", "midterm", "anon_student_id"]),
        "semester_summary": semester_summary.sort_values(["semester_raw", "midterm"]),
        "activity_corr": activity_corr,
        "change_corr": change_corr,
        "unsupported": unsupported,
    }


def set_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette=[MIDTERM_PALETTE["mid1"], MIDTERM_PALETTE["mid2"]],
        font_scale=0.95,
    )


def save_semester_overview(semester_summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    sns.barplot(
        data=semester_summary,
        x="semester_raw",
        y="mean_score_pct",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        ax=axes[0],
    )
    axes[0].set_title("Average Midterm Scores By Semester")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean score (%)")
    axes[0].legend(title="")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f", fontsize=10, padding=2)

    sns.barplot(
        data=semester_summary,
        x="semester_raw",
        y="mean_coverage_pct",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Scoring Coverage By Semester")
    axes[1].set_xlabel("Semester")
    axes[1].set_ylabel("Observed exam items scored (%)")
    axes[1].legend_.remove()
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f", fontsize=10, padding=2)

    fig.suptitle("Semester-Level Midterm Overview", y=1.02, fontsize=20)
    fig.tight_layout()
    out = FIG_DIR / "01_semester_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_score_distributions(scores: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxenplot(
        data=scores,
        x="semester_raw",
        y="score_pct",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        ax=ax,
        showfliers=False,
    )
    ax.set_title("Student Score Distributions Within Each Semester")
    ax.set_xlabel("Semester")
    ax.set_ylabel("Student midterm score (%)")
    ax.legend(title="")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = FIG_DIR / "02_score_distributions.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_activity_corr_lollipop(activity_corr: pd.DataFrame) -> Path:
    top = activity_corr.head(8)
    bottom = activity_corr.tail(8)
    chart = pd.concat([top, bottom]).drop_duplicates("feature").sort_values("within_group_rank_corr")

    fig, ax = plt.subplots(figsize=(14, 9))
    colors = ["#c0392b" if val < 0 else "#1b7f5a" for val in chart["within_group_rank_corr"]]
    ax.hlines(y=chart["feature_label"], xmin=0, xmax=chart["within_group_rank_corr"], color=colors, linewidth=3, alpha=0.75)
    ax.scatter(chart["within_group_rank_corr"], chart["feature_label"], s=120, color=colors, zorder=3)
    ax.axvline(0, color="#444444", linewidth=1.2)
    ax.set_title("Strongest Positive And Negative Activity Signals")
    ax.set_xlabel("Within-cohort rank correlation with midterm score")
    ax.set_ylabel("")

    for _, row in chart.iterrows():
        offset = 0.012 if row["within_group_rank_corr"] >= 0 else -0.012
        ha = "left" if row["within_group_rank_corr"] >= 0 else "right"
        ax.text(
            row["within_group_rank_corr"] + offset,
            row["feature_label"],
            f"{row['within_group_rank_corr']:.2f}",
            va="center",
            ha=ha,
            fontsize=10,
        )

    fig.tight_layout()
    out = FIG_DIR / "03_activity_correlation_lollipop.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_correlation_heatmap(activity_corr: pd.DataFrame) -> Path:
    rows = []
    for _, row in activity_corr.iterrows():
        family, metric = row["feature"].split("__", 1)
        if family in {"conceptcheck", "mchoice", "fillb", "dragndrop", "activecode", "parsons"} and metric in {
            "total_events",
            "active_days",
            "unique_problems",
            "avg_event_score",
            "mean_first_score",
            "mean_best_score",
            "mean_score_gain",
            "mean_events_per_problem",
        }:
            rows.append(
                {
                    "family": FAMILY_LABELS.get(family, family.title()),
                    "metric": METRIC_LABELS.get(metric, metric),
                    "corr": row["within_group_rank_corr"],
                }
            )
    matrix = pd.DataFrame(rows).pivot(index="metric", columns="family", values="corr")
    metric_order = [
        "Total events",
        "Active days",
        "Unique problems",
        "Average event score",
        "Mean first score",
        "Mean best score",
        "Mean score gain",
        "Events per problem",
    ]
    family_order = [
        "Concept Checks",
        "Multiple Choice",
        "Fill In Blank",
        "Drag and Drop",
        "ActiveCode",
        "Parsons",
    ]
    matrix = matrix.reindex(index=metric_order, columns=family_order)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Quality Beats Volume Across Most Activity Families")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    out = FIG_DIR / "04_quality_vs_volume_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_rank_scatter(features: pd.DataFrame, feature_col: str, title: str, subtitle: str, file_name: str) -> Path:
    plot_df = features[["cohort", "midterm", "score_rank_pct", feature_col]].dropna().copy()
    plot_df["feature_rank_pct"] = plot_df.groupby("cohort")[feature_col].rank(pct=True) * 100
    coeffs = np.polyfit(plot_df["feature_rank_pct"], plot_df["score_rank_pct"], deg=1)
    x_line = np.linspace(0, 100, 200)
    y_line = coeffs[0] * x_line + coeffs[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x="feature_rank_pct",
        y="score_rank_pct",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        alpha=0.28,
        s=35,
        linewidth=0,
        ax=ax,
    )
    ax.plot(x_line, y_line, color="#111111", linewidth=2.5, label="overall trend")
    ax.set_title(title)
    ax.text(2, 97, subtitle, ha="left", va="top", fontsize=11, color="#333333")
    ax.set_xlabel("Within-cohort feature percentile")
    ax.set_ylabel("Within-cohort exam score percentile")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(title="")
    fig.tight_layout()
    out = FIG_DIR / file_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_change_frame(features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in features.columns if "__" in col and pd.api.types.is_numeric_dtype(features[col])]
    wide = features[["semester_raw", "anon_student_id", "midterm", "score_pct"] + numeric_cols].set_index(
        ["semester_raw", "anon_student_id", "midterm"]
    ).unstack("midterm")
    wide.columns = [f"{metric}__{midterm}" for metric, midterm in wide.columns]
    wide = wide.reset_index()
    wide = wide.loc[wide["score_pct__mid1"].notna() & wide["score_pct__mid2"].notna()].copy()
    wide["score_change"] = wide["score_pct__mid2"] - wide["score_pct__mid1"]
    wide["parsons_total_change"] = wide["parsons__total_events__mid2"] - wide["parsons__total_events__mid1"]
    wide["parsons_first_mid1"] = wide["parsons__mean_first_score__mid1"]
    wide["baseline_rank"] = wide.groupby("semester_raw")["parsons_first_mid1"].rank(pct=True)
    wide["growth_rank"] = wide.groupby("semester_raw")["parsons_total_change"].rank(pct=True)
    wide["baseline_group"] = np.where(
        wide["baseline_rank"] <= 0.5,
        "Lower-half baseline parsons",
        "Upper-half baseline parsons",
    )
    wide["growth_group"] = pd.cut(
        wide["growth_rank"],
        bins=[0, 0.25, 0.75, 1.0],
        labels=["Low growth", "Middle growth", "High growth"],
        include_lowest=True,
    )
    return wide


def save_change_corr_plot(change_corr: pd.DataFrame) -> Path:
    chart = change_corr.head(10).sort_values("within_group_rank_corr")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("crest", n_colors=len(chart))
    ax.barh(chart["feature_label"], chart["within_group_rank_corr"], color=colors)
    ax.axvline(0, color="#444444", linewidth=1.1)
    ax.set_title("Which Mid1-To-Mid2 Behavior Changes Mattered Most?")
    ax.set_xlabel("Within-semester rank correlation with score change")
    ax.set_ylabel("")
    for value, y in zip(chart["within_group_rank_corr"], chart["feature_label"]):
        ax.text(value + 0.003, y, f"{value:.2f}", va="center", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "07_change_correlation_bars.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_parsons_change_groups(change_df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    summary = (
        change_df.dropna(subset=["baseline_group", "growth_group"])
        .groupby(["baseline_group", "growth_group"], as_index=False)
        .agg(
            n_students=("anon_student_id", "count"),
            mean_score_change=("score_change", "mean"),
            mean_mid1_pct=("score_pct__mid1", "mean"),
            mean_mid2_pct=("score_pct__mid2", "mean"),
            mean_parsons_total_change=("parsons_total_change", "mean"),
        )
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(
        data=summary,
        x="growth_group",
        y="mean_score_change",
        hue="baseline_group",
        palette=["#7fcdbb", "#2c7fb8"],
        ax=ax,
    )
    ax.axhline(0, color="#444444", linewidth=1.1)
    ax.set_title("Parsons Growth Before Mid2 Softened The Midterm-2 Drop")
    ax.set_xlabel("Change in total Parsons events before midterm 2")
    ax.set_ylabel("Average mid2 - mid1 score change (pct points)")
    ax.legend(title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=10, padding=3)
    fig.tight_layout()
    out = FIG_DIR / "08_parsons_growth_groups.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out, summary


def save_unsupported_items(unsupported: pd.DataFrame) -> Path:
    counts = unsupported["item_family"].value_counts().rename_axis("item_family").reset_index(name="count")
    counts["item_family"] = counts["item_family"].map(
        {
            "fillb": "Fill In Blank",
            "dragndrop": "Drag and Drop",
            "clickable": "Clickable",
            "shortanswer": "Short Answer",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=counts, x="count", y="item_family", hue="item_family", palette="magma", dodge=False, legend=False, ax=ax)
    ax.set_title("Unsupported Or Ambiguous Exam Items")
    ax.set_xlabel("Student-problem rows left unscored")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=10, padding=3)
    fig.tight_layout()
    out = FIG_DIR / "09_scoring_limitations.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_participation_rates(features: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    families = ["conceptcheck", "mchoice", "fillb", "dragndrop", "activecode", "parsons"]
    rows = []
    for family in families:
        col = f"{family}__unique_problems"
        use = features[["midterm", "anon_student_id", col]].copy()
        use["used"] = use[col].fillna(0).gt(0)
        tmp = use.groupby("midterm", as_index=False).agg(pct_used=("used", "mean"))
        tmp["pct_used"] *= 100
        tmp["family"] = FAMILY_LABELS.get(family, family.title())
        rows.append(tmp)
    summary = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=summary,
        x="family",
        y="pct_used",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        ax=ax,
    )
    ax.set_title("Almost Everyone Touched Every Major Activity Family")
    ax.set_xlabel("")
    ax.set_ylabel("Students with at least one pre-exam interaction (%)")
    ax.set_ylim(95, 100.5)
    ax.legend(title="")
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=10, padding=2)
    fig.tight_layout()
    out = FIG_DIR / "10_participation_rates.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out, summary


def save_typical_exposure(features: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    families = ["conceptcheck", "mchoice", "fillb", "dragndrop", "activecode", "parsons"]
    rows = []
    for family in families:
        col = f"{family}__unique_problems"
        tmp = (
            features.groupby("midterm", as_index=False)[col]
            .median()
            .rename(columns={col: "median_unique_problems"})
        )
        tmp["family"] = FAMILY_LABELS.get(family, family.title())
        rows.append(tmp)
    summary = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=summary,
        x="family",
        y="median_unique_problems",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        ax=ax,
    )
    ax.set_title("Typical Student Exposure Before Each Midterm")
    ax.set_xlabel("")
    ax.set_ylabel("Median distinct problems attempted before the exam")
    ax.legend(title="")
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=10, padding=2)
    fig.tight_layout()
    out = FIG_DIR / "11_typical_exposure.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out, summary


def save_parsons_baseline_story(features: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    plot_df = features.dropna(subset=["parsons__mean_first_score", "parsons__mean_score_gain", "score_pct"]).copy()
    plot_df["baseline_quartile"] = plot_df.groupby("cohort")["parsons__mean_first_score"].transform(
        lambda s: pd.qcut(
            s.rank(method="first"),
            4,
            labels=["Q1 lowest first-try", "Q2", "Q3", "Q4 highest first-try"],
        )
    )
    summary = (
        plot_df.groupby("baseline_quartile", observed=False, as_index=False)
        .agg(
            n_students=("anon_student_id", "count"),
            mean_parsons_first=("parsons__mean_first_score", "mean"),
            mean_parsons_gain=("parsons__mean_score_gain", "mean"),
            mean_exam_score=("score_pct", "mean"),
            mean_parsons_events=("parsons__total_events", "mean"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    sns.barplot(
        data=summary,
        x="baseline_quartile",
        y="mean_parsons_gain",
        hue="baseline_quartile",
        palette="crest",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Students Who Started Lower Improved More")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Average Parsons score gain")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", fontsize=10, padding=3)

    sns.barplot(
        data=summary,
        x="baseline_quartile",
        y="mean_exam_score",
        hue="baseline_quartile",
        palette="flare",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("But They Still Scored Lower On Midterms")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Average midterm score (%)")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f", fontsize=10, padding=3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("Why Parsons Score Gain Can Look Negative In Correlations", y=1.03, fontsize=19)
    fig.tight_layout()
    out = FIG_DIR / "12_parsons_baseline_story.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out, summary


def save_effort_efficiency_map(features: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    plot_df = features[
        ["cohort", "anon_student_id", "score_pct", "parsons__total_events", "parsons__mean_first_score"]
    ].dropna().copy()
    plot_df["effort_pct"] = plot_df.groupby("cohort")["parsons__total_events"].rank(pct=True) * 100
    plot_df["efficiency_pct"] = plot_df.groupby("cohort")["parsons__mean_first_score"].rank(pct=True) * 100
    plot_df["score_pctile"] = plot_df.groupby("cohort")["score_pct"].rank(pct=True) * 100
    plot_df["profile"] = np.select(
        [
            (plot_df["effort_pct"] > 50) & (plot_df["efficiency_pct"] > 50),
            (plot_df["effort_pct"] <= 50) & (plot_df["efficiency_pct"] > 50),
            (plot_df["effort_pct"] > 50) & (plot_df["efficiency_pct"] <= 50),
        ],
        [
            "High effort, high efficiency",
            "Low effort, high efficiency",
            "High effort, low efficiency",
        ],
        default="Low effort, low efficiency",
    )
    profile_summary = (
        plot_df.groupby("profile", as_index=False)
        .agg(
            n_students=("anon_student_id", "count"),
            mean_exam_score=("score_pct", "mean"),
            median_exam_score=("score_pct", "median"),
        )
        .sort_values("mean_exam_score", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10.5, 8))
    scatter = ax.scatter(
        plot_df["effort_pct"],
        plot_df["efficiency_pct"],
        c=plot_df["score_pctile"],
        cmap="viridis",
        alpha=0.32,
        s=36,
        linewidths=0,
    )
    ax.axvline(50, color="#444444", linestyle="--", linewidth=1.2)
    ax.axhline(50, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_title("Parsons Effort Versus Parsons Efficiency")
    ax.set_xlabel("Within-cohort percentile for Parsons activity volume")
    ax.set_ylabel("Within-cohort percentile for Parsons first-try quality")
    ax.text(5, 95, "Low effort,\nhigh efficiency", ha="left", va="top", fontsize=11, color="#1f4e5f")
    ax.text(95, 95, "High effort,\nhigh efficiency", ha="right", va="top", fontsize=11, color="#1f4e5f")
    ax.text(5, 5, "Low effort,\nlow efficiency", ha="left", va="bottom", fontsize=11, color="#7a3b3b")
    ax.text(95, 5, "High effort,\nlow efficiency", ha="right", va="bottom", fontsize=11, color="#7a3b3b")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Within-cohort midterm score percentile")
    fig.tight_layout()
    out = FIG_DIR / "13_effort_vs_efficiency.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out, profile_summary


def save_profile_outcomes(profile_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=profile_summary,
        y="profile",
        x="mean_exam_score",
        hue="profile",
        palette="viridis",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Efficiency Separates Student Profiles More Than Effort Alone")
    ax.set_xlabel("Average midterm score (%)")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=10, padding=3)
    fig.tight_layout()
    out = FIG_DIR / "14_student_profiles.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def load_or_build_timing_summary(base_dir: Path = BASE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = OUT_DIR / "cramming_bin_summary.csv"
    metrics_path = OUT_DIR / "cramming_student_metrics.csv"
    if summary_path.exists() and metrics_path.exists():
        return pd.read_csv(summary_path), pd.read_csv(metrics_path)

    scores = pd.read_parquet(
        OUT_DIR / "student_midterm_scores.parquet",
        columns=["semester_raw", "anon_student_id", "midterm", "exam_start", "score_pct"],
    )
    attempts = pd.read_parquet(
        base_dir / "analysis7_outputs" / "midterm_attempts.parquet",
        columns=["session_id"],
    )
    exam_sessions = set(attempts["session_id"].unique().tolist())
    raw = pd.read_parquet(
        base_dir / "combined_raw.parquet",
        columns=["time", "anon_student_id", "session_id", "selection", "semester"],
    ).rename(columns={"semester": "semester_raw"})
    selections = [
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
        "selectquestion",
    ]
    raw = raw.loc[raw["selection"].isin(selections)].copy()
    raw = raw.loc[~raw["session_id"].isin(exam_sessions)].copy()
    raw["event_day"] = raw["time"].dt.floor("D")

    pre = raw[["semester_raw", "anon_student_id", "event_day"]].merge(
        scores[["semester_raw", "anon_student_id", "midterm", "exam_start", "score_pct"]],
        on=["semester_raw", "anon_student_id"],
        how="inner",
    )
    pre["days_before"] = (pre["exam_start"].dt.floor("D") - pre["event_day"]).dt.days
    pre = pre.loc[(pre["days_before"] >= 1) & (pre["days_before"] <= 28)].copy()
    pre["cohort"] = pre["semester_raw"].astype(str) + "_" + pre["midterm"].astype(str)
    pre["score_group"] = pre.groupby("cohort")["score_pct"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    )
    pre = pre.loc[pre["score_group"].isin(["Q1 lowest", "Q4 highest"])].copy()
    pre["timing_bin"] = pd.cut(
        pre["days_before"],
        bins=[0, 3, 7, 14, 28],
        labels=["1-3 days before", "4-7 days before", "8-14 days before", "15-28 days before"],
        include_lowest=True,
    )

    student_bin = (
        pre.groupby(["semester_raw", "anon_student_id", "midterm", "score_group", "timing_bin"], observed=False)
        .size()
        .rename("events")
        .reset_index()
    )
    student_totals = (
        student_bin.groupby(["semester_raw", "anon_student_id", "midterm", "score_group"], as_index=False)["events"]
        .sum()
        .rename(columns={"events": "total_events_28d"})
    )
    student_bin = student_bin.merge(
        student_totals,
        on=["semester_raw", "anon_student_id", "midterm", "score_group"],
        how="left",
    )
    student_bin["share"] = student_bin["events"] / student_bin["total_events_28d"]
    summary = (
        student_bin.groupby(["score_group", "timing_bin"], observed=False, as_index=False)
        .agg(mean_share=("share", "mean"))
    )
    summary["mean_share"] *= 100

    student_day = (
        pre.groupby(["semester_raw", "anon_student_id", "midterm", "score_group", "days_before"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    daily_totals = (
        student_day.groupby(["semester_raw", "anon_student_id", "midterm", "score_group"], as_index=False)["events"]
        .sum()
        .rename(columns={"events": "total_events_28d"})
    )
    student_day = student_day.merge(
        daily_totals,
        on=["semester_raw", "anon_student_id", "midterm", "score_group"],
        how="left",
    )
    student_day["share"] = student_day["events"] / student_day["total_events_28d"]
    metrics = (
        student_day.groupby(["semester_raw", "anon_student_id", "midterm", "score_group"], as_index=False)
        .agg(
            active_days=("days_before", "nunique"),
            last3_share=("share", lambda s: s[student_day.loc[s.index, "days_before"] <= 3].sum()),
            last7_share=("share", lambda s: s[student_day.loc[s.index, "days_before"] <= 7].sum()),
            concentration_hhi=("share", lambda s: (s**2).sum()),
        )
    )

    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return summary, metrics


def load_or_build_pipeline_summary(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    summary_path = OUT_DIR / "pipeline_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)

    try:
        import pyarrow.parquet as pq

        raw_event_rows = pq.ParquetFile(base_dir / "combined_raw.parquet").metadata.num_rows
    except Exception:
        raw_event_rows = len(pd.read_parquet(base_dir / "combined_raw.parquet", columns=["time"]))

    actual_attempt_rows = len(
        pd.read_parquet(base_dir / "analysis7_outputs" / "midterm_attempts.parquet", columns=["session_id"])
    )
    exam_problem_rows = len(
        pd.read_parquet(OUT_DIR / "student_midterm_problem_scores.parquet", columns=["problem_name"])
    )
    student_midterm_rows = len(
        pd.read_parquet(OUT_DIR / "student_midterm_scores.parquet", columns=["anon_student_id"])
    )
    feature_rows = len(
        pd.read_parquet(OUT_DIR / "student_midterm_activity_features.parquet", columns=["anon_student_id"])
    )

    summary = pd.DataFrame(
        [
            {
                "step": 1,
                "label": "Raw Runestone log events",
                "count": raw_event_rows,
                "count_display": f"{raw_event_rows:,}",
                "description": "All anonymized ebook log rows across reading, lecture, practice, and exam activity.",
            },
            {
                "step": 2,
                "label": "Timed actual midterm attempts",
                "count": actual_attempt_rows,
                "count_display": f"{actual_attempt_rows:,}",
                "description": "Student exam windows inferred from timed start and finish markers in the log.",
            },
            {
                "step": 3,
                "label": "Student-exam problem scores",
                "count": exam_problem_rows,
                "count_display": f"{exam_problem_rows:,}",
                "description": "Observed exam problem rows scored from clear correctness signals or high-confidence inferred keys.",
            },
            {
                "step": 4,
                "label": "Student midterm score rows",
                "count": student_midterm_rows,
                "count_display": f"{student_midterm_rows:,}",
                "description": "One reconstructed midterm row per student, semester, and exam.",
            },
            {
                "step": 5,
                "label": "Student pre-exam feature rows",
                "count": feature_rows,
                "count_display": f"{feature_rows:,}",
                "description": "Pre-exam activity summaries used to compare students within the same semester-midterm cohort.",
            },
        ]
    )
    summary.to_csv(summary_path, index=False)
    return summary


def load_or_build_practice_midterm_summary(base_dir: Path = BASE_DIR) -> dict[str, pd.DataFrame]:
    student_path = OUT_DIR / "practice_midterm_student_summary.csv"
    attempt_path = OUT_DIR / "practice_midterm_attempt_scores.csv"
    group_path = OUT_DIR / "practice_midterm_group_summary.csv"
    cohort_path = OUT_DIR / "practice_midterm_cohort_summary.csv"
    corr_path = OUT_DIR / "practice_midterm_correlation_summary.csv"
    if all(path.exists() for path in [student_path, attempt_path, group_path, cohort_path, corr_path]):
        return {
            "student": pd.read_csv(student_path),
            "attempt": pd.read_csv(attempt_path),
            "group": pd.read_csv(group_path),
            "cohort": pd.read_csv(cohort_path),
            "corr": pd.read_csv(corr_path),
        }

    actual = pd.read_parquet(
        OUT_DIR / "student_midterm_scores.parquet",
        columns=["semester_raw", "anon_student_id", "midterm", "exam_start", "score_pct"],
    )
    raw = pd.read_parquet(
        base_dir / "combined_raw.parquet",
        columns=["time", "anon_student_id", "session_id", "problem_name", "selection", "action", "semester"],
    ).rename(columns={"semester": "semester_raw"})

    markers = raw.loc[raw["selection"].eq("timedExam")].copy()
    markers["midterm"] = markers["problem_name"].map(parse_midterm_from_name)
    markers["clean_name"] = markers["problem_name"].map(clean_problem_name)
    markers = markers.loc[
        markers["midterm"].notna() & markers["clean_name"].str.contains("practice", na=False)
    ].copy()

    attempt_keys = ["semester_raw", "anon_student_id", "session_id", "midterm"]
    starts = (
        markers.loc[markers["action"].eq("start")]
        .groupby(attempt_keys)["time"]
        .min()
        .rename("practice_start")
    )
    finishes = (
        markers.loc[markers["action"].eq("finish")]
        .groupby(attempt_keys)["time"]
        .max()
        .rename("practice_end")
    )
    practice_names = (
        markers.groupby(attempt_keys)["problem_name"]
        .agg(lambda s: " | ".join(sorted(set(map(str, s)))))
        .rename("practice_name")
    )
    practice_attempts = pd.concat([starts, finishes, practice_names], axis=1).reset_index()
    practice_attempts = practice_attempts.merge(
        actual[["semester_raw", "anon_student_id", "midterm", "exam_start"]],
        on=["semester_raw", "anon_student_id", "midterm"],
        how="inner",
    )
    practice_attempts = practice_attempts.loc[
        practice_attempts["practice_start"] < practice_attempts["exam_start"]
    ].copy()
    practice_attempts["finished_before_exam"] = (
        practice_attempts["practice_end"].notna()
        & (practice_attempts["practice_end"] < practice_attempts["exam_start"])
    )
    practice_attempts["attempt_minutes"] = (
        practice_attempts["practice_end"] - practice_attempts["practice_start"]
    ).dt.total_seconds() / 60

    started_summary = (
        practice_attempts.groupby(["semester_raw", "anon_student_id", "midterm"], as_index=False)
        .agg(
            timed_practice_starts=("practice_name", "nunique"),
            timed_practice_finishes=("finished_before_exam", "sum"),
        )
    )

    finished_attempts = practice_attempts.loc[practice_attempts["finished_before_exam"]].copy()
    direct_events = raw.loc[
        raw["selection"].isin(["mChoice", "parsons", "hparsonsAnswer", "unittest"])
    ].copy()
    direct_events["event_score"] = np.nan
    mchoice_mask = direct_events["selection"].eq("mChoice")
    parsons_mask = direct_events["selection"].eq("parsons")
    hparsons_mask = direct_events["selection"].eq("hparsonsAnswer")
    unittest_mask = direct_events["selection"].eq("unittest")
    direct_events.loc[mchoice_mask, "event_score"] = direct_events.loc[mchoice_mask, "action"].map(parse_mchoice_score)
    direct_events.loc[parsons_mask, "event_score"] = direct_events.loc[parsons_mask, "action"].map(parse_parsons_score)
    direct_events.loc[hparsons_mask, "event_score"] = direct_events.loc[hparsons_mask, "action"].map(parse_hparsons_score)
    direct_events.loc[unittest_mask, "event_score"] = direct_events.loc[unittest_mask, "action"].map(parse_unittest_pct)
    direct_events["family"] = np.select(
        [
            direct_events["selection"].eq("mChoice"),
            direct_events["selection"].eq("unittest"),
            direct_events["selection"].isin(["parsons", "hparsonsAnswer"]),
        ],
        ["mchoice", "activecode", "parsons"],
        default="other",
    )

    labeled = direct_events.merge(
        finished_attempts[
            [
                "semester_raw",
                "anon_student_id",
                "session_id",
                "midterm",
                "practice_start",
                "practice_end",
                "practice_name",
            ]
        ],
        on=["semester_raw", "anon_student_id", "session_id"],
        how="inner",
    )
    labeled = labeled.loc[
        labeled["time"].between(labeled["practice_start"], labeled["practice_end"], inclusive="both")
    ].copy()

    problem_rows = []
    group_cols = attempt_keys + ["practice_name", "problem_name"]
    for keys, group in labeled.sort_values(group_cols + ["time"]).groupby(group_cols):
        semester_raw, anon_student_id, session_id, midterm, practice_name, problem_name = keys
        family = group["family"].mode().iloc[0]
        values = group["event_score"].dropna()
        if family == "mchoice":
            practice_item_score = values.iloc[-1] if not values.empty else np.nan
        else:
            practice_item_score = values.max() if not values.empty else np.nan
        problem_rows.append(
            {
                "semester_raw": semester_raw,
                "anon_student_id": anon_student_id,
                "session_id": session_id,
                "midterm": midterm,
                "practice_name": practice_name,
                "problem_name": problem_name,
                "family": family,
                "practice_item_score": practice_item_score,
            }
        )
    problem_scores = pd.DataFrame(problem_rows)

    if problem_scores.empty:
        attempt_scores = pd.DataFrame(
            columns=attempt_keys + ["practice_name", "practice_direct_items", "practice_direct_score_pct"]
        )
    else:
        attempt_scores = (
            problem_scores.groupby(attempt_keys + ["practice_name"], as_index=False)
            .agg(
                practice_direct_items=("problem_name", "nunique"),
                practice_direct_score_pct=("practice_item_score", "mean"),
            )
            .sort_values(["semester_raw", "anon_student_id", "midterm", "practice_name"])
        )
        attempt_scores["practice_direct_score_pct"] *= 100

    best_scores = (
        attempt_scores.sort_values(
            ["semester_raw", "anon_student_id", "midterm", "practice_direct_score_pct"]
        )
        .groupby(["semester_raw", "anon_student_id", "midterm"], as_index=False)
        .agg(
            scored_practice_attempts=("practice_name", "nunique"),
            best_practice_direct_score_pct=("practice_direct_score_pct", "max"),
            latest_practice_direct_score_pct=("practice_direct_score_pct", "last"),
            median_practice_direct_items=("practice_direct_items", "median"),
        )
    )

    student_summary = (
        actual.merge(
            started_summary,
            on=["semester_raw", "anon_student_id", "midterm"],
            how="left",
        )
        .merge(
            best_scores,
            on=["semester_raw", "anon_student_id", "midterm"],
            how="left",
        )
        .copy()
    )
    for col in ["timed_practice_starts", "timed_practice_finishes", "scored_practice_attempts"]:
        student_summary[col] = student_summary[col].fillna(0).astype(int)
    student_summary["did_timed_practice"] = student_summary["timed_practice_starts"].gt(0)
    student_summary["finished_timed_practice"] = student_summary["timed_practice_finishes"].gt(0)
    student_summary["practice_group"] = np.select(
        [
            student_summary["timed_practice_starts"].eq(0),
            student_summary["timed_practice_finishes"].gt(0),
        ],
        ["No timed practice", "Finished timed practice"],
        default="Started only",
    )
    student_summary["cohort"] = (
        student_summary["semester_raw"].astype(str) + "_" + student_summary["midterm"].astype(str)
    )
    student_summary["exam_score_rank_pct"] = (
        student_summary.groupby("cohort")["score_pct"].rank(pct=True) * 100
    )
    student_summary["practice_score_rank_pct"] = (
        student_summary.groupby("cohort")["best_practice_direct_score_pct"].rank(pct=True) * 100
    )
    student_summary["practice_quality_band"] = np.select(
        [
            student_summary["practice_score_rank_pct"].le(25),
            student_summary["practice_score_rank_pct"].gt(75),
        ],
        ["Bottom quartile", "Top quartile"],
        default="Middle half",
    )
    student_summary.loc[
        student_summary["best_practice_direct_score_pct"].isna(), "practice_quality_band"
    ] = pd.NA

    group_summary = (
        student_summary.groupby(["midterm", "practice_group"], as_index=False)
        .agg(
            n_student_midterms=("anon_student_id", "count"),
            mean_actual_score=("score_pct", "mean"),
            median_actual_score=("score_pct", "median"),
        )
        .sort_values(["midterm", "practice_group"])
    )
    cohort_summary = (
        student_summary.groupby(["semester_raw", "midterm"], as_index=False)
        .agg(
            n_student_midterms=("anon_student_id", "count"),
            timed_practice_rate=("did_timed_practice", "mean"),
            finished_practice_rate=("finished_timed_practice", "mean"),
            mean_actual_score=("score_pct", "mean"),
            mean_best_practice_direct_score_pct=("best_practice_direct_score_pct", "mean"),
        )
        .sort_values(["semester_raw", "midterm"])
    )
    cohort_summary["timed_practice_rate"] *= 100
    cohort_summary["finished_practice_rate"] *= 100

    scored_only = student_summary.loc[
        student_summary["best_practice_direct_score_pct"].notna()
    ].copy()
    quartile_summary = (
        scored_only.groupby(["midterm", "practice_quality_band"], as_index=False)
        .agg(
            n_student_midterms=("anon_student_id", "count"),
            mean_actual_score=("score_pct", "mean"),
        )
        .sort_values(["midterm", "practice_quality_band"])
    )

    finished_vs_none = (
        student_summary.groupby(["cohort", "practice_group"], as_index=False)
        .agg(mean_actual_score=("score_pct", "mean"))
        .pivot(index="cohort", columns="practice_group", values="mean_actual_score")
    )
    finish_gap = np.nan
    if {"Finished timed practice", "No timed practice"}.issubset(set(finished_vs_none.columns)):
        finish_gap = (
            finished_vs_none["Finished timed practice"] - finished_vs_none["No timed practice"]
        ).mean()

    corr_summary = pd.DataFrame(
        [
            {
                "metric": "overall_within_cohort_rank_corr",
                "value": within_group_rank_corr_generic(
                    student_summary, "cohort", "best_practice_direct_score_pct", "score_pct"
                ),
                "description": "Within-cohort rank correlation between best timed practice score and actual midterm score.",
            },
            {
                "metric": "mid1_within_cohort_rank_corr",
                "value": within_group_rank_corr_generic(
                    student_summary.loc[student_summary["midterm"].eq("mid1")],
                    "cohort",
                    "best_practice_direct_score_pct",
                    "score_pct",
                ),
                "description": "Same correlation using only midterm 1 rows.",
            },
            {
                "metric": "mid2_within_cohort_rank_corr",
                "value": within_group_rank_corr_generic(
                    student_summary.loc[student_summary["midterm"].eq("mid2")],
                    "cohort",
                    "best_practice_direct_score_pct",
                    "score_pct",
                ),
                "description": "Same correlation using only midterm 2 rows.",
            },
            {
                "metric": "finish_vs_none_mean_cohort_gap",
                "value": finish_gap,
                "description": "Average within-cohort score gap between students who finished a timed practice exam and students who did no timed practice.",
            },
            {
                "metric": "practice_quality_top_vs_bottom_gap",
                "value": within_group_quartile_gap_generic(
                    student_summary, "cohort", "best_practice_direct_score_pct", "score_pct"
                ),
                "description": "Average within-cohort score gap between top and bottom timed-practice score quartiles.",
            },
            {
                "metric": "finished_attempt_rows",
                "value": float(len(finished_attempts)),
                "description": "Finished timed practice attempts observed before the real exam.",
            },
            {
                "metric": "median_direct_items_per_finished_attempt",
                "value": float(attempt_scores["practice_direct_items"].median()) if not attempt_scores.empty else np.nan,
                "description": "Median number of directly scoreable items observed in a finished timed practice attempt.",
            },
        ]
    )

    student_summary.to_csv(student_path, index=False)
    attempt_scores.to_csv(attempt_path, index=False)
    group_summary.to_csv(group_path, index=False)
    cohort_summary.to_csv(cohort_path, index=False)
    corr_summary.to_csv(corr_path, index=False)
    quartile_summary.to_csv(OUT_DIR / "practice_midterm_quartile_summary.csv", index=False)
    return {
        "student": student_summary,
        "attempt": attempt_scores,
        "group": group_summary,
        "cohort": cohort_summary,
        "corr": corr_summary,
    }


def save_practice_midterm_groups(
    student_summary: pd.DataFrame, cohort_summary: pd.DataFrame
) -> Path:
    practice_order = ["No timed practice", "Started only", "Finished timed practice"]
    plot_df = (
        student_summary.groupby(["midterm", "practice_group"], as_index=False)
        .agg(mean_actual_score=("score_pct", "mean"))
        .copy()
    )
    plot_df["practice_group"] = pd.Categorical(
        plot_df["practice_group"], categories=practice_order, ordered=True
    )

    rate_df = cohort_summary.melt(
        id_vars=["semester_raw", "midterm"],
        value_vars=["timed_practice_rate", "finished_practice_rate"],
        var_name="rate_type",
        value_name="rate_pct",
    )
    rate_df["rate_type"] = rate_df["rate_type"].map(
        {
            "timed_practice_rate": "Started timed practice",
            "finished_practice_rate": "Finished timed practice",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=plot_df,
        x="practice_group",
        y="mean_actual_score",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        ax=axes[0],
    )
    axes[0].set_title("Actual Midterm Scores By Timed Practice Group")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Average actual midterm score (%)")
    axes[0].legend(title="")
    axes[0].tick_params(axis="x", rotation=15)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f", fontsize=10, padding=2)

    sns.barplot(
        data=rate_df,
        x="semester_raw",
        y="rate_pct",
        hue="rate_type",
        palette=["#7ec8c5", "#1d7874"],
        ax=axes[1],
    )
    axes[1].set_title("How Common Were Timed Practice Midterms?")
    axes[1].set_xlabel("Semester")
    axes[1].set_ylabel("Student-midterms with timed practice (%)")
    axes[1].legend(title="")
    axes[1].tick_params(axis="x", rotation=35)

    fig.suptitle("Timed Practice Midterms: Participation And Outcome", y=1.03, fontsize=19)
    fig.tight_layout()
    out = FIG_DIR / "17_practice_midterm_groups.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_practice_midterm_quality(
    student_summary: pd.DataFrame, corr_summary: pd.DataFrame
) -> Path:
    scored_only = student_summary.loc[
        student_summary["best_practice_direct_score_pct"].notna()
    ].copy()
    quartile_summary = (
        scored_only.groupby(["midterm", "practice_quality_band"], as_index=False)
        .agg(mean_actual_score=("score_pct", "mean"))
        .copy()
    )
    quartile_order = ["Bottom quartile", "Middle half", "Top quartile"]
    quartile_summary["practice_quality_band"] = pd.Categorical(
        quartile_summary["practice_quality_band"], categories=quartile_order, ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(
        data=scored_only,
        x="practice_score_rank_pct",
        y="exam_score_rank_pct",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        alpha=0.28,
        s=50,
        ax=axes[0],
    )
    corr_value = corr_summary.loc[
        corr_summary["metric"].eq("overall_within_cohort_rank_corr"), "value"
    ].iloc[0]
    axes[0].set_title(f"Practice Quality Tracks Real Exam Performance (r≈{corr_value:.2f})")
    axes[0].set_xlabel("Within-cohort percentile for best timed practice score")
    axes[0].set_ylabel("Within-cohort percentile for actual midterm score")
    axes[0].legend(title="")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)

    sns.barplot(
        data=quartile_summary,
        x="practice_quality_band",
        y="mean_actual_score",
        hue="midterm",
        palette=MIDTERM_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Higher Timed Practice Scores Meant Better Real Exam Scores")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Average actual midterm score (%)")
    axes[1].legend(title="")
    axes[1].tick_params(axis="x", rotation=15)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f", fontsize=10, padding=2)

    fig.suptitle(
        "Timed Practice Midterms: Quality Mattered More Than Just Opening The Exam",
        y=1.03,
        fontsize=19,
    )
    fig.tight_layout()
    out = FIG_DIR / "18_practice_midterm_quality.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_cramming_consistency(timing_summary: pd.DataFrame, timing_metrics: pd.DataFrame) -> Path:
    summary = timing_summary.loc[timing_summary["score_group"].isin(["Q1 lowest", "Q4 highest"])].copy()
    summary = summary.dropna(subset=["mean_share"]).copy()
    summary["score_group"] = summary["score_group"].astype(str)
    metrics = timing_metrics.loc[timing_metrics["score_group"].isin(["Q1 lowest", "Q4 highest"])].copy()
    metrics["score_group"] = metrics["score_group"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(
        data=summary,
        x="timing_bin",
        y="mean_share",
        hue="score_group",
        palette=["#c0392b", "#1b9e77"],
        ax=axes[0],
    )
    axes[0].set_title("Where Did Activity Fall In The Last 28 Days?")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Average share of pre-exam activity (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="")

    sns.boxplot(
        data=metrics,
        x="score_group",
        y="active_days",
        hue="score_group",
        palette=["#c0392b", "#1b9e77"],
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("How Many Separate Days Did Students Work?")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Active days in last 28 days")

    fig.suptitle("Timing Differences Exist, But They Are Smaller Than Quality Differences", y=1.02, fontsize=19)
    fig.tight_layout()
    out = FIG_DIR / "15_cramming_vs_consistency.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_learning_progression(features: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    plot_df = features.copy()
    plot_df["exam_group"] = plot_df.groupby("cohort")["score_pct"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 lowest exam", "Q2", "Q3", "Q4 highest exam"])
    )
    plot_df = plot_df.loc[plot_df["exam_group"].isin(["Q1 lowest exam", "Q4 highest exam"])].copy()
    families = ["parsons", "activecode", "mchoice", "fillb"]
    rows = []
    for family in families:
        tmp = (
            plot_df.groupby("exam_group", observed=False, as_index=False)
            .agg(
                first=(f"{family}__mean_first_score", "mean"),
                last=(f"{family}__mean_last_score", "mean"),
            )
        )
        tmp["family"] = FAMILY_LABELS.get(family, family.title())
        rows.append(tmp)
    summary = pd.concat(rows, ignore_index=True)
    long = summary.melt(
        id_vars=["exam_group", "family"],
        value_vars=["first", "last"],
        var_name="stage",
        value_name="score",
    )
    long["stage"] = long["stage"].map({"first": "First observed score", "last": "Last observed score"})
    long = long.loc[long["exam_group"].isin(["Q1 lowest exam", "Q4 highest exam"])].copy()
    long["exam_group"] = long["exam_group"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes = axes.ravel()
    family_order = ["Parsons", "ActiveCode", "Multiple Choice", "Fill In Blank"]
    for ax, family in zip(axes, family_order):
        sub = long.loc[long["family"].eq(family)].copy()
        sns.lineplot(
            data=sub,
            x="stage",
            y="score",
            hue="exam_group",
            style="exam_group",
            markers=True,
            dashes=False,
            palette={"Q1 lowest exam": "#c0392b", "Q4 highest exam": "#1b9e77"},
            ax=ax,
        )
        ax.set_title(family)
        ax.set_xlabel("")
        ax.set_ylabel("Average practice score")
        ax.legend(title="", loc="best")
        ax.set_ylim(0, max(1.0, float(sub["score"].max()) + 0.05))
    fig.suptitle("All Students Improve, But Stronger Exam Groups Start Higher", y=1.02, fontsize=19)
    fig.tight_layout()
    out = FIG_DIR / "16_learning_progression.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out, summary


def render_notebook(figures: dict[str, Path], frames: dict[str, pd.DataFrame], parsons_summary: pd.DataFrame) -> None:
    sem = frames["semester_summary"].copy()
    scores = frames["scores"].copy()
    activity = frames["activity_corr"].copy()
    change = frames["change_corr"].copy()
    pipeline_summary = frames["pipeline_summary"].copy()
    practice_student = frames["practice_student"].copy()
    practice_group = frames["practice_group"].copy()
    practice_corr = frames["practice_corr"].copy()

    diff = sem.pivot(index=["semester_raw", "semester"], columns="midterm", values="mean_score_pct").reset_index()
    diff["mid2_minus_mid1"] = diff["mid2"] - diff["mid1"]
    avg_drop = diff["mid2_minus_mid1"].mean()
    worst_row = diff.loc[diff["mid2_minus_mid1"].idxmin()]

    parsons_row = activity.loc[activity["feature"].eq("parsons__mean_first_score")].iloc[0]
    activecode_row = activity.loc[activity["feature"].eq("activecode__mean_best_score")].iloc[0]
    concept_events_row = activity.loc[activity["feature"].eq("conceptcheck__total_events")].iloc[0]
    concept_unique_row = activity.loc[activity["feature"].eq("conceptcheck__unique_problems")].iloc[0]
    parsons_gain_row = activity.loc[activity["feature"].eq("parsons__mean_score_gain")].iloc[0]
    change_top = change.iloc[0]
    participation = frames["participation_summary"]
    exposure = frames["exposure_summary"]

    low_base = parsons_summary.loc[parsons_summary["baseline_group"].eq("Lower-half baseline parsons")].copy()
    if not low_base.empty:
        growth_gap = low_base.sort_values("growth_group")["mean_score_change"].iloc[-1] - low_base.sort_values("growth_group")["mean_score_change"].iloc[0]
    else:
        growth_gap = np.nan

    parsons_mid1_rate = participation.loc[
        participation["family"].eq("Parsons") & participation["midterm"].eq("mid1"),
        "pct_used",
    ].iloc[0]
    concept_mid1_rate = participation.loc[
        participation["family"].eq("Concept Checks") & participation["midterm"].eq("mid1"),
        "pct_used",
    ].iloc[0]
    parsons_mid1_median = exposure.loc[
        exposure["family"].eq("Parsons") & exposure["midterm"].eq("mid1"),
        "median_unique_problems",
    ].iloc[0]
    parsons_mid2_median = exposure.loc[
        exposure["family"].eq("Parsons") & exposure["midterm"].eq("mid2"),
        "median_unique_problems",
    ].iloc[0]
    activecode_mid1_median = exposure.loc[
        exposure["family"].eq("ActiveCode") & exposure["midterm"].eq("mid1"),
        "median_unique_problems",
    ].iloc[0]
    activecode_mid2_median = exposure.loc[
        exposure["family"].eq("ActiveCode") & exposure["midterm"].eq("mid2"),
        "median_unique_problems",
    ].iloc[0]
    parsons_baseline_story = frames["parsons_baseline_story"]
    lowest_q = parsons_baseline_story.loc[parsons_baseline_story["baseline_quartile"].eq("Q1 lowest first-try")].iloc[0]
    highest_q = parsons_baseline_story.loc[parsons_baseline_story["baseline_quartile"].eq("Q4 highest first-try")].iloc[0]
    profile_summary = frames["profile_summary"]
    best_profile = profile_summary.iloc[0]
    worst_profile = profile_summary.iloc[-1]
    progression_summary = frames["progression_summary"]
    parsons_prog_low = progression_summary.loc[
        progression_summary["family"].eq("Parsons") & progression_summary["exam_group"].eq("Q1 lowest exam")
    ].iloc[0]
    parsons_prog_high = progression_summary.loc[
        progression_summary["family"].eq("Parsons") & progression_summary["exam_group"].eq("Q4 highest exam")
    ].iloc[0]
    timing_summary = frames["timing_summary"]
    timing_metrics = frames["timing_metrics"]
    q1_last7 = timing_metrics.loc[timing_metrics["score_group"].eq("Q1 lowest"), "last7_share"].mean() * 100
    q4_last7 = timing_metrics.loc[timing_metrics["score_group"].eq("Q4 highest"), "last7_share"].mean() * 100
    q1_days = timing_metrics.loc[timing_metrics["score_group"].eq("Q1 lowest"), "active_days"].mean()
    q4_days = timing_metrics.loc[timing_metrics["score_group"].eq("Q4 highest"), "active_days"].mean()
    practice_corr_value = practice_corr.loc[
        practice_corr["metric"].eq("overall_within_cohort_rank_corr"), "value"
    ].iloc[0]
    practice_finish_gap = practice_corr.loc[
        practice_corr["metric"].eq("finish_vs_none_mean_cohort_gap"), "value"
    ].iloc[0]
    practice_quality_gap = practice_corr.loc[
        practice_corr["metric"].eq("practice_quality_top_vs_bottom_gap"), "value"
    ].iloc[0]
    practice_finished_attempts = practice_corr.loc[
        practice_corr["metric"].eq("finished_attempt_rows"), "value"
    ].iloc[0]
    practice_mid1_finished = practice_group.loc[
        practice_group["midterm"].eq("mid1") & practice_group["practice_group"].eq("Finished timed practice"),
        "mean_actual_score",
    ].iloc[0]
    practice_mid1_none = practice_group.loc[
        practice_group["midterm"].eq("mid1") & practice_group["practice_group"].eq("No timed practice"),
        "mean_actual_score",
    ].iloc[0]
    practice_mid2_finished = practice_group.loc[
        practice_group["midterm"].eq("mid2") & practice_group["practice_group"].eq("Finished timed practice"),
        "mean_actual_score",
    ].iloc[0]
    practice_mid2_none = practice_group.loc[
        practice_group["midterm"].eq("mid2") & practice_group["practice_group"].eq("No timed practice"),
        "mean_actual_score",
    ].iloc[0]

    intro = f"""# Analysis 8: Midterm Scores, Activity Patterns, and Visual Findings

This notebook is the visual version of the analysis. It uses the saved outputs in `analysis8_outputs` and is written for an instructor who wants to understand how students used the Runestone ebook and what parts of that usage were most closely associated with midterm outcomes.

## Big takeaways

- **Midterm 2 was harder almost everywhere.** Across all semesters with both exams, the average `mid2 - mid1` shift was **{avg_drop:.1f} percentage points**, and the largest drop was **{worst_row['semester_raw']} ({worst_row['mid2_minus_mid1']:.1f})**.
- **Parsons first-try quality was the strongest single signal.** Students in the top quartile of `Parsons mean first score` scored about **{parsons_row['top_quartile_minus_bottom_quartile_pct_points']:.1f} points higher** on the midterm than students in the bottom quartile of the same semester-midterm cohort.
- **Breadth beat raw clicking for concept checks.** `conceptcheck__unique_problems` was positive (**{concept_unique_row['within_group_rank_corr']:.2f}**), while `conceptcheck__total_events` was slightly negative (**{concept_events_row['within_group_rank_corr']:.2f}**).
- **Big practice gains were not automatically “good news.”** `parsons__mean_score_gain` was strongly negative (**{parsons_gain_row['within_group_rank_corr']:.2f}**), which suggests large gains often reflected weaker starting points rather than stronger final mastery.
- **Behavior changes before midterm 2 mattered, but less than baseline quality.** The strongest change feature was `{clean_feature_label(change_top['feature'])}` at **{change_top['within_group_rank_corr']:.2f}**, which is real but much smaller than the top baseline-quality signals.
- **Efficiency seems to matter more than sheer effort.** Students with high Parsons efficiency scored much better than low-efficiency students, even when both groups showed high effort.
- **Timed practice midterms added a useful readiness signal.** Finishing a timed practice exam before the real exam was associated with about **{practice_finish_gap:.1f} more real-exam points** within the same cohort, and timed-practice quality itself had a positive within-cohort relationship of about **{practice_corr_value:.2f}**.
"""

    methodology = """## How To Read These Figures

- Every correlation is computed **within semester-midterm cohorts** first, then combined, so the plots are not just picking up that one semester was easier than another.
- The scatter plots use **percentile ranks within each cohort** on both axes. A point near `(80, 80)` means a student is around the 80th percentile on both the feature and the exam score within their own cohort.
- Score reconstruction comes from the `analysis7` exam windows plus event-level correctness signals. Ambiguous `fillb`, `dragndrop`, `clickable`, and `shortanswer` rows are shown explicitly in the limitations section instead of being force-scored.
- The timed practice-midterm analysis is intentionally conservative. It uses only **timed practice windows that finished before the real exam**, and its score is based on directly scoreable practice items like multiple choice, Parsons checks, and unittest-backed coding items.
"""

    glossary = """## Plain-Language Glossary

- **Parsons puzzle**: a programming problem where students arrange mixed-up code blocks into the correct order. In the logs, `parsonsMove` records moves and `parsons` records graded checks.
- **Parsons first-try quality**: for each Parsons problem a student worked before the exam, I looked at the **first time the student checked their answer**. If that first check was correct, that problem contributes a high value; if it was incorrect, it contributes a low value. The student’s feature is the average across all of those problems. In classroom terms, this is a measure of whether students can set up a correct solution quickly rather than after lots of trial-and-error.
- **ActiveCode**: executable code cells in the ebook. The feature `ActiveCode mean best score` uses the best unit-test score the student reached on each coding problem before the exam.
- **Concept checks**: the `selectquestion` family in the logs. These appear to be the course’s regular non-exam multiple-choice concept questions, including the kinds of questions used in lecture and reading checks.
- **Total events**: every logged interaction, such as a click, move, run, or answer submission. This is a volume measure.
- **Unique problems**: the number of distinct problems a student touched. This is a breadth measure.
- **Active days**: the number of different calendar days on which the student used that activity family before the exam. This is a spacing measure.
- **Mean score gain**: the student’s last observed score on a practice problem minus their first observed score on that same problem, averaged across problems. A negative correlation here does **not** mean improvement is bad. It usually means students who began weaker had more room to improve.
- **Pre-exam**: all activity that happened before the start of the relevant midterm window. `mid1` features use behavior before midterm 1, and `mid2` features use behavior before midterm 2.
"""

    teacher_note = """## Teacher Interpretation Notes

- These visuals are best used to identify **patterns worth acting on**, not to claim causation.
- A **positive correlation** means students who show more of that pattern also tend to earn stronger midterm scores.
- A **negative correlation** often means the feature is acting as a marker of struggle. For example, many Parsons checks or large score gains can mean a student needed a lot of retries before they understood the problem.
- For teaching decisions, the safest interpretation is: if a feature reflects **clean early understanding** or **broad, consistent practice**, it is a promising signal. If a feature reflects **heavy rework**, it may help flag students who need support earlier.
"""

    connect_the_dots = f"""## Connecting The Dots

- **Why does “did the activity or not” barely matter?** Because in this course, the major activity families were already close to universal. Once nearly everyone has touched Parsons, ActiveCode, concept checks, and multiple choice, the interesting differences are about *quality*, *breadth*, and *consistency*.
- **Why are quality metrics stronger than raw counts?** A student who gets practice items mostly right on the first graded check is probably showing fluency. A student with a huge number of moves, checks, or events may instead be showing confusion, persistence, or both. Those are not the same instructional signal.
- **Why can score gain be negative even if practice helps?** Because gain is strongly shaped by starting point. In this dataset, students in the lowest Parsons first-try quartile improved by about **{lowest_q['mean_parsons_gain']:.2f}** on average, while students in the highest quartile improved by only **{highest_q['mean_parsons_gain']:.2f}**. But the lower-starting group still averaged only **{lowest_q['mean_exam_score']:.1f}%** on the midterm versus **{highest_q['mean_exam_score']:.1f}%** for the highest-starting group. That means improvement is real, but it often reflects students climbing out of a deeper hole rather than overtaking stronger peers.
- **Why might total concept-check events be weak or slightly negative?** A likely reason is that raw clicks mix together productive engagement and repeated revisiting caused by uncertainty. In contrast, the number of distinct concept-check problems and the number of days students engaged with them are more clearly positive, which fits the idea that broad, spaced exposure is healthier than repeated clicking on the same material.
- **What about cramming?** The timing story is not as simple as “strong students never cram.” In fact, strong students also do a sizable share of their activity near the exam. The difference seems to be that final-week activity by itself is not enough to explain performance; students still separate mostly on quality and efficiency.
- **Why is midterm 2 lower than midterm 1 almost everywhere?** The log data alone cannot prove the cause. Plausible explanations include more difficult later content, a harder exam, cumulative course load, or a broader skill mix on the second exam. The consistent cross-semester pattern makes it worth discussing as a structural course issue, not just a one-term anomaly.
"""

    pipeline_story = "## High-Level Pipeline\n\n" + "\n".join(
        [
            f"{int(row.step)}. **{row.label} ({row.count_display})**: {row.description}"
            for row in pipeline_summary.itertuples(index=False)
        ]
    ) + (
        "\n\nThis section is here to make the later plots more trustworthy. The report does not jump straight from raw clicks to conclusions. "
        "It first identifies real exam windows, reconstructs student-level midterm scores, then measures pre-exam behavior and compares students within the same semester-midterm cohorts."
    )

    load_cell = """from pathlib import Path
import pandas as pd

base_dir = Path.cwd()
out_dir = base_dir / "analysis8_outputs"

scores = pd.read_parquet(out_dir / "student_midterm_scores.parquet")
features = pd.read_parquet(out_dir / "student_midterm_activity_features.parquet")
semester_summary = pd.read_csv(out_dir / "semester_midterm_score_summary.csv")
activity_corr = pd.read_csv(out_dir / "activity_correlations.csv")
change_corr = pd.read_csv(out_dir / "activity_change_correlations.csv")
unsupported = pd.read_csv(out_dir / "unsupported_or_ambiguous_exam_items.csv")
"""

    def img_cell(title: str, body: str, fig_path: Path) -> nbf.NotebookNode:
        rel = fig_path.relative_to(BASE_DIR).as_posix()
        return nbf.v4.new_markdown_cell(f"## {title}\n\n{body}\n\n![{title}]({rel})")

    cells = [
        nbf.v4.new_markdown_cell(intro),
        nbf.v4.new_markdown_cell(pipeline_story),
        nbf.v4.new_markdown_cell(methodology),
        nbf.v4.new_markdown_cell(glossary),
        nbf.v4.new_markdown_cell(teacher_note),
        nbf.v4.new_markdown_cell(connect_the_dots),
        nbf.v4.new_code_cell(load_cell),
        img_cell(
            "Who Actually Used Each Activity Type?",
            f"This figure answers a basic course-design question first: before each midterm, what percentage of students had **any** interaction with each major activity family? The answer is: almost everyone. For example, Parsons usage is already about **{parsons_mid1_rate:.1f}%** before `mid1`, and concept checks are about **{concept_mid1_rate:.1f}%**. That means simple participation is not the key differentiator in this course. Students generally encountered all of these tools. The more useful question for an instructor is **how** they used them: quickly or slowly, broadly or narrowly, on one day or over many days, with immediate correctness or repeated rework. Teacher takeaway: because exposure is nearly universal, interventions should focus less on getting students to click once and more on helping them practice successfully and consistently.",
            figures["participation"],
        ),
        img_cell(
            "How Much Practice Did A Typical Student Do?",
            f"This chart shows the median number of distinct problems a student had touched before each exam. It gives a sense of what “normal” exposure looked like in this course. A typical student had worked around **{parsons_mid1_median:.0f} Parsons problems** before `mid1` and **{parsons_mid2_median:.0f}** before `mid2`. For ActiveCode, the median rose from about **{activecode_mid1_median:.0f}** to **{activecode_mid2_median:.0f}**. The main point is that students had substantial contact with these activity types well before the exams. Teacher takeaway: if an instructor wants to improve outcomes, the strongest leverage may be improving the *quality* of these interactions rather than simply adding more instances of the same activity.",
            figures["exposure"],
        ),
        img_cell(
            "Semester Overview",
            "This figure has two panels. The top panel shows the average reconstructed midterm score in each semester, split into `mid1` and `mid2`. The bottom panel shows how much of each exam could be scored from observable item-level signals. The main story is that `mid2` sits below `mid1` in every semester, while scoring coverage stays high enough that this pattern is unlikely to be a data artifact. For a teacher, this suggests the second exam may consistently be harder, later material may be more challenging, or cumulative fatigue may be setting in by that point of the semester. Teacher takeaway: if this pattern matches classroom experience, the instructor may want to inspect the content, pacing, and support structures leading into the second midterm.",
            figures["semester_overview"],
        ),
        img_cell(
            "Score Distributions By Semester",
            "These boxen plots show the spread of student scores within each semester and midterm, not just the average. This helps separate two ideas: whether one exam was lower on average, and whether it was also more spread out. Several `mid2` cohorts shift downward as a whole rather than just adding a few low outliers, which supports the idea that the second midterm was broadly tougher. In other words, this does not look like a story driven by a tiny subgroup alone. Teacher takeaway: if an exam shift affects the whole distribution, the response may need to be course-wide, such as earlier review, more scaffolding, or adjusted expectations for the later unit.",
            figures["score_distributions"],
        ),
        img_cell(
            "Strongest Activity Signals",
            "This lollipop chart ranks the strongest positive and negative relationships between pre-exam behavior and midterm performance. Positive values mean the feature tends to go with stronger midterm scores; negative values mean the feature tends to go with weaker scores. The standout result is Parsons quality: first-try and average Parsons correctness dominate the chart. The surprising negative result is that large score gains on practice are often a weakness signal, because students who started much lower had more room to improve. This is one of the most important interpretation points for the whole notebook: high effort is not always the same thing as high preparedness. Teacher takeaway: repeated retries can be used as an early-warning signal for struggle, while clean early success can be used as a signal of readiness.",
            figures["activity_corr"],
        ),
        img_cell(
            "Quality Versus Volume Heatmap",
            "This heatmap groups the correlations by activity family and metric type. Read across a row to compare one metric, or down a column to see the overall pattern for a family. Warm cells are positive, cool cells are negative. The main pattern is very consistent: quality metrics such as `mean first score`, `mean best score`, and `average event score` are more informative than raw event counts. Concept checks are especially interesting because `unique problems` and `active days` help, while simple `total events` do not. In practical terms, touching more concept-check questions across more days seems healthier than clicking the same question family many times in a concentrated burst. Teacher takeaway: this argues for distributed practice and broad coverage rather than just more activity volume.",
            figures["quality_heatmap"],
        ),
        img_cell(
            "Parsons First-Try Quality Versus Midterm Outcome",
            f"This scatter plot puts both the Parsons feature and the exam score into within-cohort percentiles, so each dot compares a student against peers from the same semester and same midterm. The upward slope is steep: moving from the bottom quartile to the top quartile of Parsons first-try quality is associated with about **{parsons_row['top_quartile_minus_bottom_quartile_pct_points']:.1f} extra midterm points**. That is one of the strongest effects anywhere in the notebook. To make this concrete for a teacher: students who can assemble a reasonable Parsons solution and get it correct on the first graded check tend to be much more exam-ready than students who need many checks before arriving at the answer. Teacher takeaway: Parsons puzzles may be especially useful not just as practice, but as a diagnostic. Students who repeatedly miss the first check could be flagged for support before the exam.",
            figures["parsons_scatter"],
        ),
        img_cell(
            "Effort Versus Efficiency",
            f"This plot compares two different ideas that are easy to confuse. The horizontal axis is **effort**: how much Parsons activity a student generated relative to classmates in the same cohort. The vertical axis is **efficiency**: how strong that student’s Parsons first-try quality was relative to classmates. The color shows the student’s midterm score percentile. The key pattern is that high-efficiency students tend to do well whether their effort is low or high, while low-efficiency students tend to struggle even when their effort is high. Teacher takeaway: a lot of activity is not automatically a sign of understanding. A student can be working very hard and still need help. This makes Parsons logs useful for distinguishing productive practice from repeated struggle.",
            figures["effort_efficiency"],
        ),
        img_cell(
            "Student Behavior Profiles",
            f"This bar chart turns the effort-efficiency map into four simple profiles. The strongest profile is **{best_profile['profile']}**, with an average midterm score around **{best_profile['mean_exam_score']:.1f}%**. The weakest profile is **{worst_profile['profile']}**, at about **{worst_profile['mean_exam_score']:.1f}%**. The especially interesting comparison is between **high effort, high efficiency** and **high effort, low efficiency**: effort alone does not close the gap. Teacher takeaway: if an instructor wants an early warning flag, high-effort students with low efficiency may be one of the most important groups to catch, because they are engaged but still not converting effort into understanding.",
            figures["profiles"],
        ),
        img_cell(
            "ActiveCode Best Practice Quality Versus Midterm Outcome",
            f"This plot repeats the same percentile idea for `ActiveCode mean best score`. The relationship is still clearly positive, but not as dominant as Parsons. Students in the top quartile of this feature outscored bottom-quartile students by about **{activecode_row['top_quartile_minus_bottom_quartile_pct_points']:.1f} points**, which is still a meaningful gap. Since ActiveCode problems are executable and testable code, this result is consistent with the idea that students who can get coding practice to pass unit tests before the exam are more likely to perform well on the exam too. Teacher takeaway: coding practice quality matters, but the stronger Parsons signal suggests that code-tracing and code-assembly fluency may be especially important in this course.",
            figures["activecode_scatter"],
        ),
        img_cell(
            "Timed Practice Midterms: Participation And Completion",
            f"This figure introduces a separate but very practical question for instructors: what happened when students used the course’s timed practice midterms before the real exam? The left panel compares actual midterm scores for three groups: students who did no timed practice, students who opened a timed practice exam but did not finish one before the real exam, and students who finished one. The strongest group is the finishers. On `mid1`, students who finished a timed practice exam averaged about **{practice_mid1_finished:.1f}%** on the real exam, compared with **{practice_mid1_none:.1f}%** for students who did none. On `mid2`, the same comparison is about **{practice_mid2_finished:.1f}%** versus **{practice_mid2_none:.1f}%**. The right panel shows that timed practice usage varied by semester, which matters because it reminds us this is an opportunity signal, not something every cohort received equally. Teacher takeaway: simply opening a practice exam is not the same as working through it. Completion appears to be the more meaningful readiness marker.",
            figures["practice_groups"],
        ),
        img_cell(
            "Timed Practice Midterms: Quality As A Readiness Check",
            f"This figure goes one step further. Among students with a finished timed practice exam, it asks whether stronger performance on the directly scoreable practice items also lined up with stronger real-exam performance. The left scatter uses within-cohort percentiles, and the relationship is clearly positive at about **{practice_corr_value:.2f}**. The right panel makes that easier to read: students in the **top timed-practice quality quartile** scored about **{practice_quality_gap:.1f} points** higher on the real exam than students in the bottom quartile of the same cohort. This practice score is conservative: it only uses directly scoreable items inside the timed practice windows, and it is based on roughly **{practice_finished_attempts:.0f} finished timed practice attempts** overall. Teacher takeaway: practice midterms appear most useful as a diagnostic when students actually complete them and when instructors pay attention to how well students do, not just whether the practice link was opened.",
            figures["practice_quality"],
        ),
        img_cell(
            "Cramming Versus Consistency",
            f"This figure checks a common hypothesis directly. The left panel shows how top and bottom exam quartiles distributed their activity over the last 28 days before the exam. The right panel shows how many separate days they were active in that window. The result is more nuanced than a simple anti-cramming story: top students still did a lot of work in the last week, averaging about **{q4_last7:.1f}%** of their last-28-day activity there, compared with **{q1_last7:.1f}%** for the bottom quartile. Their average number of active days was also only slightly higher, about **{q4_days:.1f}** versus **{q1_days:.1f}**. Teacher takeaway: last-week review seems normal for everyone. Timing by itself is not the main separator here; practice quality remains the stronger signal.",
            figures["cramming"],
        ),
        img_cell(
            "What Changed From Midterm 1 To Midterm 2?",
            "This chart looks only at students who have both midterms in the same semester and asks which behavior changes line up with score changes. The bars are noticeably smaller than the baseline-quality plots above, which is an important result by itself: what students already know going into the exam matters more than short-term behavior changes. Still, the best change signals lean toward increasing Parsons engagement before `mid2`. Teacher takeaway: last-minute changes in behavior can help somewhat, but the bigger instructional opportunity may be building stronger habits and understanding earlier in the term rather than relying on short recovery windows before the second exam.",
            figures["change_corr"],
        ),
        img_cell(
            "Parsons Growth Groups Before Midterm 2",
            f"This grouped bar chart breaks students into lower- versus upper-baseline Parsons groups and then compares low, middle, and high growth in total Parsons activity before `mid2`. Everyone still tends to drop from `mid1` to `mid2`, but higher Parsons growth usually softens that drop. In the lower-baseline group, moving from low to high Parsons growth changed the average `mid2 - mid1` drop by about **{growth_gap:.1f} points**. That is one of the clearest change-based signals in the notebook, even though it is still smaller than the baseline-quality effects. Teacher takeaway: Parsons practice appears especially promising as a support tool for students who are not already strong. It may not erase the gap, but it may help reduce the size of the decline on the second exam.",
            figures["parsons_growth"],
        ),
        img_cell(
            "Learning Progression",
            f"This figure compares first observed and last observed practice scores for the lowest and highest exam quartiles across several activity families. The big pattern is that both groups improve, but the stronger exam group usually starts higher and stays higher. For Parsons, for example, the lowest exam quartile starts around **{parsons_prog_low['first']:.2f}** and ends around **{parsons_prog_low['last']:.2f}**, while the highest exam quartile starts higher at about **{parsons_prog_high['first']:.2f}** and ends higher at about **{parsons_prog_high['last']:.2f}**. Teacher takeaway: the issue is not that struggling students fail to improve at all. Many do improve. The issue is that early gaps remain meaningful by exam time, which argues for support that begins earlier rather than only after students have already fallen behind.",
            figures["progression"],
        ),
        img_cell(
            "Why Improvement Margin Can Mislead",
            f"This figure directly addresses one of the main interpretation traps in the data. Students in the **lowest** Parsons first-try quartile improved the most on Parsons practice, averaging about **{lowest_q['mean_parsons_gain']:.2f}** points of gain, while students in the **highest** quartile improved less, around **{highest_q['mean_parsons_gain']:.2f}**. But the lower-starting group still ended with lower exam scores overall. This is why `Parsons mean score gain` shows up as a negative correlation: it is often measuring *how far behind a student started*, not whether practice was useless. Teacher takeaway: a large improvement margin can actually be a sign that a student needed substantial recovery. That is useful information, but it should be interpreted as a support signal rather than a failure of the activity.",
            figures["parsons_baseline_story"],
        ),
        img_cell(
            "Scoring Limitations",
            "This final figure shows which exam item families were left unscored because the logs were ambiguous or did not contain a stable correctness signal. Most of the unresolved rows are `fillb`, not multiple choice, Parsons, or ActiveCode. That means the main headline relationships are being driven by the activity families with the cleanest signals, while the murkier item types are kept separate instead of being guessed. Teacher takeaway: you can place more confidence in the trends involving Parsons, ActiveCode, and multiple-choice practice than in anything tied to the noisier item types.",
            figures["limitations"],
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    with (BASE_DIR / "analysis8.ipynb").open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)


def build_visual_report(base_dir: Path = BASE_DIR) -> dict[str, Path]:
    set_style()
    frames = prep_frames(load_outputs())
    frames["pipeline_summary"] = load_or_build_pipeline_summary(base_dir)

    participation_fig, participation_summary = save_participation_rates(frames["features"])
    exposure_fig, exposure_summary = save_typical_exposure(frames["features"])
    parsons_story_fig, parsons_story_summary = save_parsons_baseline_story(frames["features"])
    effort_fig, profile_summary = save_effort_efficiency_map(frames["features"])
    profile_fig = save_profile_outcomes(profile_summary)
    timing_summary, timing_metrics = load_or_build_timing_summary(base_dir)
    practice_frames = load_or_build_practice_midterm_summary(base_dir)
    cramming_fig = save_cramming_consistency(timing_summary, timing_metrics)
    progression_fig, progression_summary = save_learning_progression(frames["features"])
    frames["participation_summary"] = participation_summary
    frames["exposure_summary"] = exposure_summary
    frames["parsons_baseline_story"] = parsons_story_summary
    frames["profile_summary"] = profile_summary
    frames["timing_summary"] = timing_summary
    frames["timing_metrics"] = timing_metrics
    frames["progression_summary"] = progression_summary
    frames["practice_student"] = practice_frames["student"]
    frames["practice_group"] = practice_frames["group"]
    frames["practice_cohort"] = practice_frames["cohort"]
    frames["practice_corr"] = practice_frames["corr"]

    participation_summary.to_csv(OUT_DIR / "participation_summary.csv", index=False)
    exposure_summary.to_csv(OUT_DIR / "typical_exposure_summary.csv", index=False)
    profile_summary.to_csv(OUT_DIR / "student_profile_summary.csv", index=False)
    progression_summary.to_csv(OUT_DIR / "learning_progression_summary.csv", index=False)

    figures = {
        "participation": participation_fig,
        "exposure": exposure_fig,
        "semester_overview": save_semester_overview(frames["semester_summary"]),
        "score_distributions": save_score_distributions(frames["scores"]),
        "activity_corr": save_activity_corr_lollipop(frames["activity_corr"]),
        "quality_heatmap": save_correlation_heatmap(frames["activity_corr"]),
        "parsons_scatter": save_rank_scatter(
            frames["features"],
            "parsons__mean_first_score",
            "Parsons First-Try Quality Is The Clearest Single Signal",
            "Both axes are within-cohort percentiles; the line is the overall trend.",
            "05_parsons_first_score_scatter.png",
        ),
        "activecode_scatter": save_rank_scatter(
            frames["features"],
            "activecode__mean_best_score",
            "ActiveCode Best Scores Also Track Midterm Strength",
            "This effect is positive, but not as steep as the Parsons pattern.",
            "06_activecode_best_score_scatter.png",
        ),
        "change_corr": save_change_corr_plot(frames["change_corr"]),
        "parsons_baseline_story": parsons_story_fig,
        "effort_efficiency": effort_fig,
        "profiles": profile_fig,
        "cramming": cramming_fig,
        "progression": progression_fig,
        "practice_groups": save_practice_midterm_groups(
            practice_frames["student"], practice_frames["cohort"]
        ),
        "practice_quality": save_practice_midterm_quality(
            practice_frames["student"], practice_frames["corr"]
        ),
    }
    change_df = build_change_frame(frames["features"])
    figures["parsons_growth"], parsons_summary = save_parsons_change_groups(change_df)
    figures["limitations"] = save_unsupported_items(frames["unsupported"])

    render_notebook(figures, frames, parsons_summary)
    return figures


if __name__ == "__main__":
    build_visual_report(BASE_DIR)
