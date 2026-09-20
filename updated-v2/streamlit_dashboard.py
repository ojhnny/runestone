from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "analysis_outputs"

PRIMARY = "#1d7874"
ACCENT = "#f4c95d"
DARK = "#0f4c5c"
RED = "#c0392b"
GREEN = "#1b9e77"
LIGHT_BG = "#f7fbfc"
CARD_BG = "#ffffff"
SOFT = "#edf7f6"

st.set_page_config(
    page_title="Stakeholder Report: Interactive eBook Engagement",
    page_icon="📘",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7fbfc 0%, #eef5f6 100%);
        color: #17323b !important;
        font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }
    .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span {
        color: #17323b !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.72);
        border-right: 1px solid rgba(29,120,116,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #17323b !important;
    }
    .hero {
        background: linear-gradient(120deg, #0f4c5c 0%, #1d7874 55%, #f4c95d 100%);
        padding: 26px 30px;
        border-radius: 18px;
        margin-bottom: 16px;
        color: white !important;
        box-shadow: 0 10px 28px rgba(15, 76, 92, 0.16);
    }
    .hero h1, .hero p, .hero strong, .hero span, .hero div {
        color: white !important;
    }
    .callout {
        background: #fff8e7;
        border-left: 6px solid #f4c95d;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 12px 0 18px 0;
    }
    .note {
        background: #edf7f6;
        border-left: 6px solid #1d7874;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 12px 0 18px 0;
    }
    .section-intro {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(29,120,116,0.10);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0 18px 0;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 10px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: rgba(255,255,255,0.75);
        border-radius: 999px;
        border: 1px solid rgba(29,120,116,0.10);
        padding: 10px 18px;
        color: #17323b !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #1d7874 !important;
        color: white !important;
        border-color: #1d7874 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] p {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _optional_csv(name: str) -> pd.DataFrame:
    path = OUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "scores": pd.read_parquet(OUT_DIR / "student_midterm_scores.parquet"),
        "features": pd.read_parquet(OUT_DIR / "student_midterm_activity_features.parquet"),
        "semester_summary": pd.read_csv(OUT_DIR / "semester_midterm_score_summary.csv"),
        "activity_corr": pd.read_csv(OUT_DIR / "activity_correlations.csv"),
        "change_corr": pd.read_csv(OUT_DIR / "activity_change_correlations.csv"),
        "participation": pd.read_csv(OUT_DIR / "participation_summary.csv"),
        "exposure": pd.read_csv(OUT_DIR / "typical_exposure_summary.csv"),
        "profiles": pd.read_csv(OUT_DIR / "student_profile_summary.csv"),
        "progression": pd.read_csv(OUT_DIR / "learning_progression_summary.csv"),
        "timing_bins": pd.read_csv(OUT_DIR / "cramming_bin_summary.csv"),
        "timing_metrics": pd.read_csv(OUT_DIR / "cramming_student_metrics.csv"),
        "unsupported": pd.read_csv(OUT_DIR / "unsupported_or_ambiguous_exam_items.csv"),
        "pipeline": pd.read_csv(OUT_DIR / "pipeline_summary.csv"),
        "practice_student": pd.read_csv(OUT_DIR / "practice_midterm_student_summary.csv"),
        "practice_group": pd.read_csv(OUT_DIR / "practice_midterm_group_summary.csv"),
        "practice_cohort": pd.read_csv(OUT_DIR / "practice_midterm_cohort_summary.csv"),
        "practice_corr": pd.read_csv(OUT_DIR / "practice_midterm_correlation_summary.csv"),
        "sql_corr": _optional_csv("activity_correlations_with_sql.csv"),
        "regression": _optional_csv("parsons_regression.csv"),
        "at_risk": _optional_csv("at_risk_cv.csv"),
        "at_risk_coefs": _optional_csv("at_risk_coefs.csv"),
        "headlines": _optional_csv("headline_metrics.csv"),
        "sql_semester": _optional_csv("sql_events_by_semester.csv"),
    }


def style_plot(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
        font={"family": "Avenir Next, Helvetica, Arial, sans-serif", "color": "#17323b"},
        title_font={"color": "#17323b"},
        legend={"font": {"color": "#17323b"}},
        hoverlabel={"font": {"color": "#17323b"}},
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(
        tickfont={"color": "#17323b"},
        title_font={"color": "#17323b"},
        gridcolor="rgba(23,50,59,0.08)",
        zerolinecolor="rgba(23,50,59,0.12)",
    )
    fig.update_yaxes(
        tickfont={"color": "#17323b"},
        title_font={"color": "#17323b"},
        gridcolor="rgba(23,50,59,0.08)",
        zerolinecolor="rgba(23,50,59,0.12)",
    )
    fig.update_annotations(font={"color": "#17323b"})
    if height is not None:
        fig.update_layout(height=height)
    return fig


def show_plot(fig: go.Figure, *, use_container_width: bool = True) -> None:
    st.plotly_chart(fig, use_container_width=use_container_width, theme=None)


def show_metric_cards(cards: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(cards))
    for col, (label, value, body) in zip(cols, cards):
        col.markdown(
            f"""
            <div style="background:{CARD_BG}; border:1px solid rgba(29,120,116,0.10); border-radius:16px; padding:16px 18px; box-shadow:0 3px 18px rgba(15,76,92,0.06); min-height:122px;">
              <div style="font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em; color:{PRIMARY}; margin-bottom:8px;">{label}</div>
              <div style="font-size:1.9rem; font-weight:700; color:{DARK}; margin-bottom:8px;">{value}</div>
              <div style="font-size:0.95rem; color:#47626a;">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def quarter_gap(corr_df: pd.DataFrame, feature: str) -> float:
    return corr_df.loc[corr_df["feature"].eq(feature), "top_quartile_minus_bottom_quartile_pct_points"].iloc[0]


def make_heatmap_df(activity_corr: pd.DataFrame) -> pd.DataFrame:
    label_map = {
        "conceptcheck": "Concept Checks",
        "mchoice": "Multiple Choice",
        "fillb": "Fill In Blank",
        "dragndrop": "Drag and Drop",
        "activecode": "ActiveCode",
        "parsons": "Parsons",
    }
    metric_map = {
        "total_events": "Total events",
        "active_days": "Active days",
        "unique_problems": "Unique problems",
        "avg_event_score": "Average event score",
        "mean_first_score": "Mean first score",
        "mean_best_score": "Mean best score",
        "mean_score_gain": "Mean score gain",
        "mean_events_per_problem": "Events per problem",
    }
    rows = []
    for _, row in activity_corr.iterrows():
        if "__" not in row["feature"]:
            continue
        family, metric = row["feature"].split("__", 1)
        if family in label_map and metric in metric_map:
            rows.append(
                {
                    "family": label_map[family],
                    "metric": metric_map[metric],
                    "corr": row["within_group_rank_corr"],
                }
            )
    return pd.DataFrame(rows)


def build_effort_efficiency(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    summary = (
        plot_df.groupby("profile", as_index=False)
        .agg(
            n_students=("anon_student_id", "count"),
            mean_exam_score=("score_pct", "mean"),
        )
        .sort_values("mean_exam_score", ascending=False)
    )
    return plot_df, summary


def build_parsons_rank_scatter(features: pd.DataFrame, feature_col: str, label: str) -> pd.DataFrame:
    plot_df = features[["cohort", "midterm", "score_pct", feature_col]].dropna().copy()
    plot_df["feature_rank"] = plot_df.groupby("cohort")[feature_col].rank(pct=True) * 100
    plot_df["score_rank"] = plot_df.groupby("cohort")["score_pct"].rank(pct=True) * 100
    plot_df["label"] = label
    return plot_df


def build_feature_quartiles(features: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    plot_df = features[["cohort", "midterm", "score_pct", feature_col]].dropna().copy()
    plot_df["feature_rank"] = plot_df.groupby("cohort")[feature_col].rank(pct=True, method="average")
    plot_df["feature_band"] = pd.cut(
        plot_df["feature_rank"],
        bins=[0, 0.25, 0.75, 1],
        labels=["Bottom quartile", "Middle half", "Top quartile"],
        include_lowest=True,
    )
    summary = (
        plot_df.groupby(["midterm", "feature_band"], observed=False, as_index=False)
        .agg(mean_exam_score=("score_pct", "mean"), n_students=("score_pct", "size"))
        .dropna(subset=["feature_band"])
    )
    return summary


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
        wide["baseline_rank"] <= 0.5, "Lower-half baseline", "Upper-half baseline"
    )
    wide["growth_group"] = pd.cut(
        wide["growth_rank"],
        bins=[0, 0.25, 0.75, 1],
        labels=["Low growth", "Middle growth", "High growth"],
        include_lowest=True,
    )
    return wide


def main() -> None:
    data = load_data()
    scores = data["scores"].copy()
    features = data["features"].copy()
    semester_summary = data["semester_summary"].copy()
    activity_corr = data["activity_corr"].copy()
    change_corr = data["change_corr"].copy()
    participation = data["participation"].copy()
    exposure = data["exposure"].copy()
    progression = data["progression"].copy()
    timing_bins = data["timing_bins"].copy()
    timing_metrics = data["timing_metrics"].copy()
    unsupported = data["unsupported"].copy()
    pipeline = data["pipeline"].copy()
    practice_student = data["practice_student"].copy()
    practice_group = data["practice_group"].copy()
    practice_cohort = data["practice_cohort"].copy()
    practice_corr = data["practice_corr"].copy()
    sql_corr = data["sql_corr"].copy()
    regression = data["regression"].copy()
    at_risk = data["at_risk"].copy()
    at_risk_coefs = data["at_risk_coefs"].copy()
    headlines = data["headlines"].copy()
    sql_semester = data["sql_semester"].copy()

    diff = semester_summary.pivot(index=["semester_raw", "semester"], columns="midterm", values="mean_score_pct").reset_index()
    diff["mid2_minus_mid1"] = diff["mid2"] - diff["mid1"]
    avg_drop = diff["mid2_minus_mid1"].mean()
    parsons_gap = quarter_gap(activity_corr, "parsons__mean_first_score")
    activecode_gap = quarter_gap(activity_corr, "activecode__mean_best_score")
    eff_df, profile_summary = build_effort_efficiency(features)
    practice_corr_value = practice_corr.loc[
        practice_corr["metric"].eq("overall_within_cohort_rank_corr"), "value"
    ].iloc[0]
    practice_finish_gap = practice_corr.loc[
        practice_corr["metric"].eq("finish_vs_none_mean_cohort_gap"), "value"
    ].iloc[0]
    practice_quality_gap = practice_corr.loc[
        practice_corr["metric"].eq("practice_quality_top_vs_bottom_gap"), "value"
    ].iloc[0]
    practice_finished_attempts = int(
        practice_corr.loc[practice_corr["metric"].eq("finished_attempt_rows"), "value"].iloc[0]
    )

    st.markdown(
        f"""
        <div class="hero">
          <h1>Interactive Stakeholder Report</h1>
          <p>Student engagement with interactive eBooks and what it appears to mean for midterm success.</p>
          <p><strong>Headline story:</strong> almost everyone used the ebook, but the students who used it more accurately and efficiently tended to perform better. Midterm 2 was also consistently harder than midterm 1 by about <strong>{avg_drop:.1f} points</strong>, and timed practice midterms added an extra readiness signal.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Navigation")
    st.sidebar.write("The report is organized into tabs across the top so you can move through the story without leaving the page.")
    st.sidebar.markdown(
        """
        <div class="note">
        <strong>Best walkthrough order:</strong><br>
        Overview → Method → Signals → Profiles → Practice → Midterm 2 → Trust
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_method, tab_signals, tab_profiles, tab_practice, tab_mid2, tab_models, tab_trust = st.tabs(
        [
            "Overview",
            "Method",
            "Signals",
            "Profiles",
            "Practice Midterms",
            "Midterm 2",
            "Models & SQL",
            "Trust & Actions",
        ]
    )

    with tab_overview:
        show_metric_cards(
            [
                ("Average Mid2 Drop", f"{avg_drop:.1f} pts", "Midterm 2 came in lower than Midterm 1 across every semester with both exams."),
                ("Parsons Gap", f"{parsons_gap:.1f} pts", "Top-quartile Parsons first-try students substantially outperformed bottom-quartile students."),
                ("Practice Finish Gap", f"{practice_finish_gap:.1f} pts", "Students who finished a timed practice exam tended to do better than students who did no timed practice."),
                ("ActiveCode Gap", f"{activecode_gap:.1f} pts", "Coding-practice quality mattered too, though not as strongly as Parsons."),
            ]
        )

        st.markdown(
            """
            <div class="section-intro">
            This app is designed as a walkthrough for a non-technical stakeholder. The main story is not “did students use the ebook?” because almost everyone did. The more useful questions are whether students used it successfully, whether some tools were stronger diagnostic signals than others, and what an instructor could do with those signals.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="callout">
            <strong>How to read this app:</strong> The goal is not to prove causation. The goal is to show which patterns are strong enough to matter for teaching decisions. The most reliable patterns in this report are about quality, efficiency, and broad practice, not just raw activity volume.
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                semester_summary,
                x="semester_raw",
                y="mean_score_pct",
                color="midterm",
                barmode="group",
                color_discrete_map={"mid1": GREEN, "mid2": RED},
                title="Average Midterm Score By Semester",
                labels={"semester_raw": "Semester", "mean_score_pct": "Mean score (%)"},
            )
            style_plot(fig, height=430)
            show_plot(fig)
        with c2:
            fig = px.box(
                scores,
                x="midterm",
                y="score_pct",
                color="midterm",
                points=False,
                color_discrete_map={"mid1": GREEN, "mid2": RED},
                title="Student Score Distribution By Midterm",
                labels={"midterm": "", "score_pct": "Student midterm score (%)"},
            )
            style_plot(fig, height=430)
            show_plot(fig)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                participation,
                x="family",
                y="pct_used",
                color="midterm",
                barmode="group",
                color_discrete_map={"mid1": GREEN, "mid2": RED},
                title="Who Used Each Activity Type Before The Exam?",
                labels={"family": "", "pct_used": "Students with at least one interaction (%)"},
            )
            style_plot(fig, height=420)
            fig.update_yaxes(range=[95, 100.5])
            show_plot(fig)
        with c4:
            fig = px.bar(
                exposure,
                x="family",
                y="median_unique_problems",
                color="midterm",
                barmode="group",
                color_discrete_map={"mid1": GREEN, "mid2": RED},
                title="How Much Practice Did A Typical Student Do?",
                labels={"family": "", "median_unique_problems": "Median distinct problems attempted"},
            )
            style_plot(fig, height=420)
            show_plot(fig)

        st.markdown(
            """
            <div class="note">
            <strong>Main message:</strong> the ebook was already used heavily by almost everyone, so the most useful distinctions are about how accurately and efficiently students practiced, not whether they clicked at all.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_method:
        show_metric_cards(
            [
                ("Raw Events", pipeline.loc[pipeline["step"].eq(1), "count_display"].iloc[0], "All anonymized Runestone events across reading, practice, lecture, and exams."),
                ("Exam Windows", pipeline.loc[pipeline["step"].eq(2), "count_display"].iloc[0], "Timed actual-midterm windows detected from start and finish markers."),
                ("Scored Exam Rows", pipeline.loc[pipeline["step"].eq(3), "count_display"].iloc[0], "Student-exam problem rows reconstructed from observable correctness signals."),
                ("Student Midterms", pipeline.loc[pipeline["step"].eq(4), "count_display"].iloc[0], "Reconstructed student-midterm score rows used throughout the report."),
            ]
        )

        st.markdown(
            """
            <div class="section-intro">
            This section is here to make the analysis easier to trust. The report does not leap from raw clicks to conclusions. It first reconstructs real exam windows, then rebuilds student-level midterm scores, then summarizes pre-exam behavior and compares students within the same semester-midterm cohort.
            </div>
            """,
            unsafe_allow_html=True,
        )

        labels = [
            f"{row['label']}<br>{row['count_display']}" for _, row in pipeline.sort_values("step").iterrows()
        ]
        fig = go.Figure(
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    label=labels,
                    color=[DARK, PRIMARY, "#2a9d8f", ACCENT, "#84a59d"],
                    pad=22,
                    thickness=22,
                    line=dict(color="rgba(15,76,92,0.18)", width=1),
                    x=[0.02, 0.25, 0.50, 0.75, 0.98],
                    y=[0.5, 0.5, 0.5, 0.5, 0.5],
                ),
                link=dict(
                    source=[0, 1, 2, 3],
                    target=[1, 2, 3, 4],
                    value=[1, 1, 1, 1],
                    color=[
                        "rgba(29,120,116,0.18)",
                        "rgba(29,120,116,0.18)",
                        "rgba(29,120,116,0.18)",
                        "rgba(244,201,93,0.25)",
                    ],
                ),
            )
        )
        fig.update_layout(title="High-Level Pipeline From Raw Logs To Teacher-Facing Insights")
        style_plot(fig, height=420)
        show_plot(fig)

        for row in pipeline.sort_values("step").itertuples(index=False):
            st.markdown(
                f"""
                <div class="note">
                <strong>Step {int(row.step)}: {row.label}</strong><br>
                {row.description}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_signals:
        st.markdown(
            """
            <div class="section-intro">
            These charts answer the most important instructional question in the project: what kinds of ebook behavior actually lined up with stronger midterm outcomes? The answer is mostly about quality, efficiency, and breadth rather than sheer activity volume.
            </div>
            """,
            unsafe_allow_html=True,
        )
        top_bottom = pd.concat([activity_corr.head(8), activity_corr.tail(8)]).drop_duplicates("feature").copy()
        top_bottom["direction"] = np.where(top_bottom["within_group_rank_corr"] >= 0, "Positive", "Negative")
        fig = px.bar(
            top_bottom.sort_values("within_group_rank_corr"),
            x="within_group_rank_corr",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Positive": GREEN, "Negative": RED},
            title="Strongest Positive And Negative Signals",
            labels={"within_group_rank_corr": "Within-cohort rank correlation", "feature": ""},
        )
        style_plot(fig, height=460)
        show_plot(fig)

        heat = make_heatmap_df(activity_corr)
        signal_left, signal_right = st.columns([1.15, 1])
        with signal_left:
            heat_fig = px.imshow(
                heat.pivot(index="metric", columns="family", values="corr"),
                color_continuous_scale="RdBu_r",
                zmin=-0.45,
                zmax=0.55,
                aspect="auto",
                title="Quality Beats Volume Across Most Activity Families",
            )
            style_plot(heat_fig, height=500)
            heat_fig.update_layout(coloraxis_colorbar_title="Correlation")
            show_plot(heat_fig)
        with signal_right:
            feature_options = {
                "Parsons first-try quality": ("parsons__mean_first_score", "Parsons"),
                "ActiveCode best score": ("activecode__mean_best_score", "ActiveCode"),
                "Concept-check breadth": ("conceptcheck__unique_problems", "Concept Checks"),
                "Multiple-choice first score": ("mchoice__mean_first_score", "Multiple Choice"),
            }
            feature_label = st.selectbox(
                "Choose one signal to inspect more closely",
                list(feature_options.keys()),
                index=0,
            )
            feature_col, point_label = feature_options[feature_label]
            scatter = build_parsons_rank_scatter(features, feature_col, point_label)
            quartiles = build_feature_quartiles(features, feature_col)

            fig = px.scatter(
                scatter,
                x="feature_rank",
                y="score_rank",
                color="midterm",
                opacity=0.26,
                title=f"{feature_label} Versus Real Exam Standing",
                labels={
                    "feature_rank": "Within-cohort percentile for selected signal",
                    "score_rank": "Within-cohort percentile for actual exam score",
                    "midterm": "",
                },
                color_discrete_map={"mid1": GREEN, "mid2": RED},
            )
            style_plot(fig, height=280)
            show_plot(fig)

            fig = px.bar(
                quartiles,
                x="feature_band",
                y="mean_exam_score",
                color="midterm",
                barmode="group",
                title="Quartile Comparison",
                labels={"feature_band": "", "mean_exam_score": "Average actual midterm score (%)"},
                color_discrete_map={"mid1": GREEN, "mid2": RED},
            )
            style_plot(fig, height=250)
            show_plot(fig)

        st.markdown(
            f"""
            <div class="callout">
            <strong>Story so far:</strong> Parsons first-try quality shows the strongest gap in the whole dataset at about <strong>{parsons_gap:.1f} points</strong>. ActiveCode quality matters too, but less strongly. Broad concept-check coverage helps; raw concept-check clicking does not tell nearly as much. The selector above lets you swap between signal types without leaving the page.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_profiles:
        st.markdown(
            """
            <div class="section-intro">
            This section connects the dots between effort, efficiency, and improvement. It helps distinguish students who are quietly ready, students who are struggling visibly, and students who are working hard but still not getting traction.
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = px.scatter(
            eff_df,
            x="effort_pct",
            y="efficiency_pct",
            color="score_pctile",
            color_continuous_scale="Viridis",
            opacity=0.32,
            title="Parsons Effort Versus Parsons Efficiency",
            labels={
                "effort_pct": "Within-cohort percentile for Parsons activity volume",
                "efficiency_pct": "Within-cohort percentile for Parsons first-try quality",
                "score_pctile": "Midterm percentile",
            },
        )
        fig.add_vline(x=50, line_dash="dash", line_color="#444444")
        fig.add_hline(y=50, line_dash="dash", line_color="#444444")
        style_plot(fig, height=500)
        show_plot(fig)

        fig = px.bar(
            profile_summary.sort_values("mean_exam_score"),
            x="mean_exam_score",
            y="profile",
            orientation="h",
            color="mean_exam_score",
            color_continuous_scale="Viridis",
            title="Efficiency Separates Profiles More Than Effort Alone",
            labels={"mean_exam_score": "Average midterm score (%)", "profile": ""},
        )
        style_plot(fig, height=420)
        fig.update_layout(coloraxis_showscale=False)
        show_plot(fig)

        prog = progression.loc[
            progression["exam_group"].isin(["Q1 lowest exam", "Q4 highest exam"])
            & progression["family"].isin(["Parsons", "ActiveCode", "Multiple Choice", "Fill In Blank"])
        ].copy()
        prog_long = prog.melt(
            id_vars=["family", "exam_group"],
            value_vars=["first", "last"],
            var_name="stage",
            value_name="score",
        )
        prog_long["stage"] = prog_long["stage"].map({"first": "First observed score", "last": "Last observed score"})
        fig = px.line(
            prog_long,
            x="stage",
            y="score",
            color="exam_group",
            facet_col="family",
            facet_col_wrap=2,
            markers=True,
            title="Both Groups Improve, But Stronger Students Start Higher",
            labels={"score": "Average practice score", "stage": "", "exam_group": ""},
            color_discrete_map={"Q1 lowest exam": RED, "Q4 highest exam": GREEN},
        )
        style_plot(fig, height=720)
        show_plot(fig)

        st.markdown(
            """
            <div class="note">
            <strong>Teacher interpretation:</strong> Some students are working very hard and still struggling. Those high-effort, low-efficiency students may be one of the most actionable groups to identify early, because they appear motivated but are not yet getting traction.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_practice:
        show_metric_cards(
            [
                ("Practice Finish Gap", f"{practice_finish_gap:.1f} pts", "Average within-cohort gain for students who finished a timed practice exam versus students who did no timed practice."),
                ("Practice Quality Corr.", f"{practice_corr_value:.2f}", "Within-cohort rank correlation between best timed practice score and the real midterm score."),
                ("Top-Bottom Practice Gap", f"{practice_quality_gap:.1f} pts", "Gap between top and bottom timed-practice score quartiles."),
                ("Finished Attempts", f"{practice_finished_attempts:,}", "Finished timed practice attempts observed before the real exam."),
            ]
        )

        st.markdown(
            """
            <div class="section-intro">
            This section focuses on the practice-midterm idea directly. The key message is that practice seems most informative when students actually complete a timed practice exam, and the quality of their observable work inside that practice window gives an additional readiness signal.
            </div>
            """,
            unsafe_allow_html=True,
        )

        group_order = ["No timed practice", "Started only", "Finished timed practice"]
        fig = px.bar(
            practice_group.assign(
                practice_group=pd.Categorical(
                    practice_group["practice_group"], categories=group_order, ordered=True
                )
            ).sort_values(["midterm", "practice_group"]),
            x="practice_group",
            y="mean_actual_score",
            color="midterm",
            barmode="group",
            color_discrete_map={"mid1": GREEN, "mid2": RED},
            title="Timed Practice Completion Was More Informative Than Just Opening The Practice Exam",
            labels={"practice_group": "", "mean_actual_score": "Average actual midterm score (%)"},
        )
        style_plot(fig, height=430)
        show_plot(fig)

        practice_scored = practice_student.loc[
            practice_student["best_practice_direct_score_pct"].notna()
        ].copy()
        fig = px.scatter(
            practice_scored,
            x="practice_score_rank_pct",
            y="exam_score_rank_pct",
            color="midterm",
            opacity=0.28,
            hover_data=["semester_raw", "best_practice_direct_score_pct", "score_pct"],
            color_discrete_map={"mid1": GREEN, "mid2": RED},
            title="Higher Timed Practice Scores Usually Meant Higher Real-Exam Standing",
            labels={
                "practice_score_rank_pct": "Within-cohort percentile for best timed practice score",
                "exam_score_rank_pct": "Within-cohort percentile for actual exam score",
                "midterm": "",
            },
        )
        style_plot(fig, height=470)
        show_plot(fig)

        rate_long = practice_cohort.melt(
            id_vars=["semester_raw", "midterm"],
            value_vars=["timed_practice_rate", "finished_practice_rate"],
            var_name="rate_type",
            value_name="rate_pct",
        )
        rate_long["rate_type"] = rate_long["rate_type"].map(
            {
                "timed_practice_rate": "Started timed practice",
                "finished_practice_rate": "Finished timed practice",
            }
        )
        fig = px.bar(
            rate_long,
            x="semester_raw",
            y="rate_pct",
            color="rate_type",
            facet_row="midterm",
            barmode="group",
            color_discrete_map={"Started timed practice": "#7ec8c5", "Finished timed practice": PRIMARY},
            title="Practice Availability And Completion Varied By Semester",
            labels={"semester_raw": "Semester", "rate_pct": "Student-midterms with timed practice (%)", "rate_type": ""},
        )
        style_plot(fig, height=620)
        show_plot(fig)

        st.markdown(
            f"""
            <div class="note">
            <strong>Teacher interpretation:</strong> this is best read as a readiness signal, not proof of causation. Students who finished timed practice midterms outperformed students with no timed practice by about <strong>{practice_finish_gap:.1f} points</strong> on average within the same cohort, and the top timed-practice quality group outperformed the bottom group by about <strong>{practice_quality_gap:.1f} points</strong>. That makes practice midterms promising both as preparation and as a diagnostic checkpoint.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_mid2:
        top_change = change_corr.head(10).sort_values("within_group_rank_corr")
        fig = px.bar(
            top_change,
            x="within_group_rank_corr",
            y="feature",
            orientation="h",
            color="within_group_rank_corr",
            color_continuous_scale="Tealgrn",
            title="Which Behavior Changes Mattered Most From Mid1 To Mid2?",
            labels={"within_group_rank_corr": "Correlation with score change", "feature": ""},
        )
        style_plot(fig, height=430)
        fig.update_layout(coloraxis_showscale=False)
        show_plot(fig)

        change_df = build_change_frame(features)
        growth_summary = (
            change_df.dropna(subset=["baseline_group", "growth_group"])
            .groupby(["baseline_group", "growth_group"], as_index=False)
            .agg(mean_score_change=("score_change", "mean"))
        )
        fig = px.bar(
            growth_summary,
            x="growth_group",
            y="mean_score_change",
            color="baseline_group",
            barmode="group",
            title="Parsons Growth Helped Most For Lower-Baseline Students",
            labels={"growth_group": "Change in Parsons activity before midterm 2", "mean_score_change": "Average mid2 - mid1 score change"},
            color_discrete_map={"Lower-half baseline": GREEN, "Upper-half baseline": RED},
        )
        style_plot(fig, height=430)
        show_plot(fig)

        timing_summary = timing_bins.loc[timing_bins["score_group"].isin(["Q1 lowest", "Q4 highest"])].dropna(subset=["mean_share"]).copy()
        fig = px.bar(
            timing_summary,
            x="timing_bin",
            y="mean_share",
            color="score_group",
            barmode="group",
            title="Top Students Also Work Near The Exam",
            labels={"timing_bin": "", "mean_share": "Average share of last-28-day activity (%)", "score_group": ""},
            color_discrete_map={"Q1 lowest": RED, "Q4 highest": GREEN},
        )
        style_plot(fig, height=430)
        show_plot(fig)

        st.markdown(
            """
            <div class="callout">
            <strong>Interpretation:</strong> Timing differences exist, but they are smaller than quality differences. Stronger students still do a lot of work in the final week. The bigger difference is that their practice tends to be cleaner and more efficient.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_models:
        if headlines.empty or regression.empty:
            st.info("Run `python -m ebook_analysis` to build the DuckDB features and models for this tab.")
        else:
            fe_coef = headlines.loc[headlines["metric"].eq("parsons_fe_coef"), "value"].iloc[0]
            fe_lo = headlines.loc[headlines["metric"].eq("parsons_fe_ci_low"), "value"].iloc[0]
            fe_hi = headlines.loc[headlines["metric"].eq("parsons_fe_ci_high"), "value"].iloc[0]
            ctrl = headlines.loc[headlines["metric"].eq("parsons_controlled_coef"), "value"].iloc[0]
            auc = headlines.loc[headlines["metric"].eq("at_risk_oof_auc"), "value"].iloc[0]
            show_metric_cards(
                [
                    ("Parsons FE", f"{fe_coef:.2f} pts", f"95% CI {fe_lo:.2f} to {fe_hi:.2f}. Cohort fixed effects."),
                    ("After effort controls", f"{ctrl:.2f} pts", "Still positive after events, active days, coverage, last-week share."),
                    ("At-risk AUC", f"{auc:.2f}", "Leave-one-semester-out model for bottom-quartile exams."),
                ]
            )
            st.markdown(
                """
                <div class="section-intro">
                This tab is the more technical follow-up. The 17.5-point quartile gap is still the instructor-facing number. The regression asks a slightly different question: after comparing students in the same semester and exam, how many midterm points move with Parsons first-try quality? The at-risk model is only a baseline. It is not a placement test.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not sql_semester.empty:
                fig = px.bar(
                    sql_semester,
                    x="semester",
                    y="n_events",
                    title="Raw ebook events by semester (DuckDB)",
                    labels={"semester": "Semester", "n_events": "Events"},
                )
                style_plot(fig, height=360)
                show_plot(fig)
            if not sql_corr.empty:
                extra = sql_corr.loc[sql_corr["feature"].str.contains("coverage__|timing__|consistency__|mix__", regex=True)].head(12)
                if not extra.empty:
                    fig = px.bar(
                        extra.sort_values("within_group_rank_corr"),
                        x="within_group_rank_corr",
                        y="feature",
                        orientation="h",
                        title="New SQL features vs midterm score (within cohort)",
                        labels={"within_group_rank_corr": "Rank correlation", "feature": ""},
                    )
                    style_plot(fig, height=420)
                    show_plot(fig)
            st.dataframe(regression, hide_index=True, use_container_width=True)
            if not at_risk.empty:
                st.caption("Leave-one-semester-out AUC. The overall_oof row is the one to quote.")
                st.dataframe(at_risk, hide_index=True, use_container_width=True)
            if not at_risk_coefs.empty:
                fig = px.bar(
                    at_risk_coefs.sort_values("coef"),
                    x="coef",
                    y="feature",
                    orientation="h",
                    title="At-risk model coefficients (sign only; AUC is the validation number)",
                    labels={"coef": "Logistic coefficient", "feature": ""},
                )
                style_plot(fig, height=420)
                show_plot(fig)
            st.markdown(
                """
                <div class="note">
                Looker Studio tables live in <code>looker_studio/</code>. The files there are aggregates. Student-level rows stay in <code>looker_studio/private/</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_trust:
        counts = unsupported["item_family"].value_counts().rename_axis("item_family").reset_index(name="count")
        fig = px.bar(
            counts,
            x="count",
            y="item_family",
            orientation="h",
            color="count",
            color_continuous_scale="Magma",
            title="Unsupported Or Ambiguous Exam Items",
            labels={"count": "Student-problem rows left unscored", "item_family": ""},
        )
        style_plot(fig, height=430)
        fig.update_layout(coloraxis_showscale=False)
        show_plot(fig)

        st.markdown(
            """
            <div class="note">
            <strong>What to trust most:</strong> The strongest findings are the ones tied to clearer signals like Parsons, ActiveCode, and multiple-choice style interactions. The murkier item families were kept separate instead of being guessed.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ### Practical recommendations
            1. Use Parsons first-check quality as an early-warning signal.
            2. Pay special attention to high-effort, low-efficiency students.
            3. Encourage broad and accurate practice rather than just more activity.
            4. Encourage students to finish timed practice midterms, not just open them.
            5. Revisit what happens before midterm 2, since that drop appears consistently across semesters.
            6. Interpret large improvement margins as possible signs of late recovery, not just success.
            """
        )


if __name__ == "__main__":
    main()
