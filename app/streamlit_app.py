"""Interactive readout of the SO-ELL results.

    make app      # or: uv run streamlit run app/streamlit_app.py

Reads only what the pipeline wrote to outputs/ (aggregate tables, safe to
share) plus data/processed/student_exam_features.parquet for the per-student
scatter, which exists only on a machine that ran the pipeline.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from so_ell import config

T = config.TABLES_DIR
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9890"
MID = {"mid1": BLUE, "mid2": ORANGE}

st.set_page_config(page_title="SO-ELL ebook engagement", layout="wide")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name)


@st.cache_data
def features() -> pd.DataFrame | None:
    path = config.PROCESSED_DIR / "student_exam_features.parquet"
    return pd.read_parquet(path) if path.exists() else None


def metric(name: str) -> float:
    h = pd.read_csv(config.OUTPUTS_DIR / "headline_metrics.csv").set_index("metric")["value"]
    return float(h[name])


if not (config.OUTPUTS_DIR / "headline_metrics.csv").exists():
    st.error("No outputs yet. Run `make pipeline` first.")
    st.stop()

st.title("What kind of ebook practice lines up with midterm scores?")
st.caption("Runestone interactive-ebook logs, SI 206 / SI 201, 8 semesters. Associations, not causal effects.")

c = st.columns(4)
c[0].metric("Log events", f"{metric('raw_events'):,.0f}")
c[1].metric("Student-exams scored", f"{metric('student_exams'):,.0f}")
c[2].metric(
    "Parsons first-try gap",
    f"{metric('parsons_quartile_gap'):.1f} pts",
    help=f"95% CI {metric('parsons_quartile_gap_ci_low'):.1f} to {metric('parsons_quartile_gap_ci_high'):.1f}",
)
c[3].metric("At-risk AUC (held-out semester)", f"{metric('at_risk_auc_logistic'):.2f}")

tabs = st.tabs(["Cohorts", "Parsons signal", "All features", "Timing & practice exams", "At-risk model", "Data & method"])

with tabs[0]:
    summary = table("cohort_score_summary.csv")
    fig = px.bar(summary, x="semester_raw", y="mean_score_pct", color="midterm", barmode="group",
                 color_discrete_map=MID, labels={"semester_raw": "semester", "mean_score_pct": "mean score (%)"},
                 hover_data=["n_students", "median_score_pct", "mean_coverage_pct"])
    st.plotly_chart(fig, use_container_width=True)
    st.write("Exam difficulty moves by semester, so every comparison in this project is made within a semester-midterm cohort.")
    st.dataframe(summary, hide_index=True)

with tabs[1]:
    q = table("feature_quartile_scores.csv")
    feat = st.selectbox("Feature", sorted(q["feature"].unique()), index=sorted(q["feature"].unique()).index("parsons__mean_first_score"))
    part = q.loc[q["feature"].eq(feat)]
    pooled = part.groupby("feature_quartile", as_index=False).apply(
        lambda g: pd.Series({"mean_score_pct": (g["mean_score_pct"] * g["n"]).sum() / g["n"].sum()}), include_groups=False
    )
    st.plotly_chart(px.bar(pooled, x="feature_quartile", y="mean_score_pct", color_discrete_sequence=[BLUE],
                           labels={"feature_quartile": "within-cohort quartile", "mean_score_pct": "mean midterm score (%)"}),
                    use_container_width=True)
    st.dataframe(table("parsons_models.csv")[["spec", "n", "coef", "ci_low", "ci_high", "p_value", "note"]], hide_index=True)
    f = features()
    if f is not None:
        f = f.dropna(subset=["parsons__mean_first_score"]).copy()
        f["cohort"] = f["semester_raw"] + "_" + f["midterm"]
        f["x"] = f.groupby("cohort")["parsons__mean_first_score"].rank(pct=True) * 100
        f["y"] = f.groupby("cohort")["score_pct"].rank(pct=True) * 100
        st.plotly_chart(px.scatter(f, x="x", y="y", color="midterm", color_discrete_map=MID, opacity=0.35, trendline="lowess",
                                   labels={"x": "Parsons first-try percentile", "y": "midterm percentile"}),
                        use_container_width=True)

with tabs[2]:
    corr = table("feature_signals.csv")
    only_sig = st.checkbox("Only features significant after FDR control (q < 0.05)", value=True)
    show = corr.loc[corr["q_value_bh"] < 0.05] if only_sig else corr
    st.dataframe(show, hide_index=True)

with tabs[3]:
    bins = table("cramming_timing_bins.csv")
    st.plotly_chart(px.bar(bins, x="timing_bin", y="mean_share_pct", color="score_group", barmode="group",
                           labels={"mean_share_pct": "% of last-28-day practice", "timing_bin": ""}),
                    use_container_width=True)
    group = table("practice_exam_groups.csv")
    st.plotly_chart(px.bar(group, x="practice_group", y="mean_exam_score", color="midterm", barmode="group",
                           color_discrete_map=MID, labels={"mean_exam_score": "mean midterm score (%)", "practice_group": ""}),
                    use_container_width=True)
    st.dataframe(table("practice_exam_signals.csv"), hide_index=True)

with tabs[4]:
    comp = table("at_risk_model_comparison.csv")
    st.dataframe(comp, hide_index=True)
    lead = table("at_risk_lead_time.csv")
    fig = px.line(lead, x="cutoff_days", y="auc", color="model", markers=True,
                  labels={"cutoff_days": "days before exam the data stops", "auc": "ROC AUC"})
    fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(table("at_risk_importance_gbm.csv"), hide_index=True)

with tabs[5]:
    st.dataframe(table("pipeline_funnel.csv"), hide_index=True)
    st.dataframe(pd.read_csv(config.QUALITY_DIR / "event_log_checks.csv"), hide_index=True)
    st.dataframe(table("unscored_items_summary.csv"), hide_index=True)
    st.dataframe(table("key_threshold_sensitivity.csv"), hide_index=True)
    st.markdown("Full method and design decisions: `docs/design_report.pdf`.")
