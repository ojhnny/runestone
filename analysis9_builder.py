from __future__ import annotations

from pathlib import Path
import re

import mistune
import nbformat as nbf
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "analysis8_outputs"
FIG_DIR = OUT_DIR / "figures"


def md_image(file_name: str, alt: str) -> str:
    rel = (FIG_DIR / file_name).relative_to(BASE_DIR).as_posix()
    return f'<div class="figure-frame"><img src="{rel}" alt="{alt}"></div>'


def load_analysis8_notes() -> dict[str, str]:
    path = BASE_DIR / "analysis8.ipynb"
    if not path.exists():
        return {}
    nb = nbf.read(path, as_version=4)
    notes: dict[str, str] = {}
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        src = cell.source.strip()
        if not src.startswith("## "):
            continue
        lines = src.splitlines()
        title = lines[0].replace("## ", "", 1).strip()
        body = "\n".join(lines[1:]).strip()
        body = re.sub(r"!\[.*?\]\(.*?\)", "", body, flags=re.DOTALL).strip()
        if body:
            notes[title] = body
    return notes


def detail_note(title: str, notes: dict[str, str]) -> str:
    text = notes.get(title, "").strip()
    if not text:
        return ""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = text.replace("\n", " ")
    return f'\n<div class="detail-note"><strong>Detailed figure note from analysis8:</strong> {text}</div>\n'


def metric_setup(text: str) -> str:
    return f'\n<div class="metric-setup"><strong>Why we looked at this metric:</strong> {text}</div>\n'


def export_standalone_html(cells: list[nbf.NotebookNode]) -> None:
    markdown = mistune.create_markdown(escape=False)
    rendered_sections = []
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        rendered = markdown(cell.source)
        rendered_sections.append(f'<section class="report-section">{rendered}</section>')

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Student Engagement with Interactive eBooks and Midterm Success</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1120;
      --bg-soft: #111827;
      --bg-card: rgba(15, 23, 42, 0.88);
      --text: #e5eef2;
      --muted: #a7b9c2;
      --line: rgba(148, 163, 184, 0.16);
      --teal: #14b8a6;
      --teal-soft: #0f766e;
      --amber: #f59e0b;
      --slate: #1f2937;
    }}
    * {{
      box-sizing: border-box;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background:
        radial-gradient(circle at top, rgba(20, 184, 166, 0.12) 0%, transparent 26%),
        radial-gradient(circle at 85% 10%, rgba(245, 158, 11, 0.10) 0%, transparent 22%),
        linear-gradient(180deg, #0b1120 0%, #0a0f1b 100%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.65;
    }}
    body {{
      min-height: 100vh;
    }}
    .report-shell {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 36px 26px 80px;
    }}
    .report-section {{
      margin: 0 0 30px 0;
    }}
    .report-section + .report-section {{
      margin-top: 18px;
    }}
    p, li {{
      color: var(--text);
      font-size: 1.03rem;
    }}
    ul, ol {{
      padding-left: 1.3rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
      color: #f8fbfc;
      line-height: 1.2;
      letter-spacing: -0.02em;
      margin-top: 0;
    }}
    h2 {{
      font-size: 2rem;
      margin-bottom: 0.9rem;
      padding-top: 0.3rem;
    }}
    h3 {{
      font-size: 1.35rem;
    }}
    strong {{
      color: #f8fbfc;
    }}
    em {{
      color: #cfe7e3;
    }}
    a {{
      color: #7dd3fc;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .report-section > p:first-of-type {{
      color: var(--muted);
    }}
    img {{
      max-width: 100%;
    }}
    hr {{
      border: none;
      border-top: 1px solid var(--line);
      margin: 2rem 0;
    }}
  </style>
</head>
<body>
  <main class="report-shell">
    {''.join(rendered_sections)}
  </main>
</body>
</html>
"""
    (BASE_DIR / "analysis9_report.html").write_text(html, encoding="utf-8")


def build_report() -> None:
    analysis8_notes = load_analysis8_notes()
    semester_summary = pd.read_csv(OUT_DIR / "semester_midterm_score_summary.csv")
    activity_corr = pd.read_csv(OUT_DIR / "activity_correlations.csv")
    change_corr = pd.read_csv(OUT_DIR / "activity_change_correlations.csv")
    participation = pd.read_csv(OUT_DIR / "participation_summary.csv")
    exposure = pd.read_csv(OUT_DIR / "typical_exposure_summary.csv")
    profiles = pd.read_csv(OUT_DIR / "student_profile_summary.csv")
    progression = pd.read_csv(OUT_DIR / "learning_progression_summary.csv")
    timing_bins = pd.read_csv(OUT_DIR / "cramming_bin_summary.csv")
    timing_metrics = pd.read_csv(OUT_DIR / "cramming_student_metrics.csv")
    unsupported = pd.read_csv(OUT_DIR / "unsupported_or_ambiguous_exam_items.csv")
    pipeline_summary = pd.read_csv(OUT_DIR / "pipeline_summary.csv")
    practice_group = pd.read_csv(OUT_DIR / "practice_midterm_group_summary.csv")
    practice_corr = pd.read_csv(OUT_DIR / "practice_midterm_correlation_summary.csv")
    scores = pd.read_parquet(OUT_DIR / "student_midterm_scores.parquet")
    midterm_summary = pd.read_csv(BASE_DIR / "analysis7_outputs" / "midterm_summary.csv")
    midterm_missing = pd.read_csv(BASE_DIR / "analysis7_outputs" / "midterm_missing_combinations.csv")
    ambiguous_names = pd.read_csv(BASE_DIR / "analysis7_outputs" / "midterm_ambiguous_problem_names.csv")

    diff = semester_summary.pivot(index=["semester_raw", "semester"], columns="midterm", values="mean_score_pct").reset_index()
    diff["mid2_minus_mid1"] = diff["mid2"] - diff["mid1"]
    avg_drop = diff["mid2_minus_mid1"].mean()
    worst_drop = diff.loc[diff["mid2_minus_mid1"].idxmin()]

    parsons_gap = activity_corr.loc[
        activity_corr["feature"].eq("parsons__mean_first_score"),
        "top_quartile_minus_bottom_quartile_pct_points",
    ].iloc[0]
    activecode_gap = activity_corr.loc[
        activity_corr["feature"].eq("activecode__mean_best_score"),
        "top_quartile_minus_bottom_quartile_pct_points",
    ].iloc[0]
    concept_unique_corr = activity_corr.loc[
        activity_corr["feature"].eq("conceptcheck__unique_problems"),
        "within_group_rank_corr",
    ].iloc[0]
    concept_total_corr = activity_corr.loc[
        activity_corr["feature"].eq("conceptcheck__total_events"),
        "within_group_rank_corr",
    ].iloc[0]
    parsons_gain_corr = activity_corr.loc[
        activity_corr["feature"].eq("parsons__mean_score_gain"),
        "within_group_rank_corr",
    ].iloc[0]
    strongest_change = change_corr.iloc[0]

    parsons_use_mid1 = participation.loc[
        participation["family"].eq("Parsons") & participation["midterm"].eq("mid1"),
        "pct_used",
    ].iloc[0]
    concept_use_mid1 = participation.loc[
        participation["family"].eq("Concept Checks") & participation["midterm"].eq("mid1"),
        "pct_used",
    ].iloc[0]
    parsons_exposure_mid1 = exposure.loc[
        exposure["family"].eq("Parsons") & exposure["midterm"].eq("mid1"),
        "median_unique_problems",
    ].iloc[0]
    parsons_exposure_mid2 = exposure.loc[
        exposure["family"].eq("Parsons") & exposure["midterm"].eq("mid2"),
        "median_unique_problems",
    ].iloc[0]
    activecode_exposure_mid1 = exposure.loc[
        exposure["family"].eq("ActiveCode") & exposure["midterm"].eq("mid1"),
        "median_unique_problems",
    ].iloc[0]
    activecode_exposure_mid2 = exposure.loc[
        exposure["family"].eq("ActiveCode") & exposure["midterm"].eq("mid2"),
        "median_unique_problems",
    ].iloc[0]

    best_profile = profiles.iloc[0]
    worst_profile = profiles.iloc[-1]
    high_effort_low_eff = profiles.loc[profiles["profile"].eq("High effort, low efficiency")].iloc[0]

    parsons_progression_low = progression.loc[
        progression["family"].eq("Parsons") & progression["exam_group"].eq("Q1 lowest exam")
    ].iloc[0]
    parsons_progression_high = progression.loc[
        progression["family"].eq("Parsons") & progression["exam_group"].eq("Q4 highest exam")
    ].iloc[0]

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
    finished_attempts = practice_corr.loc[
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

    improvement_story = (
        "Students who started weaker on Parsons improved more on Parsons practice, but they still scored lower on the exam. "
        "That means a big improvement margin is often a sign of recovery from a weaker starting point, not proof that the activity was unhelpful."
    )

    ambiguous_fillb = int(unsupported["item_family"].eq("fillb").sum())
    mean_coverage = scores["coverage_pct"].mean()
    confident_combos = len(midterm_summary)
    ambiguous_problem_name_count = len(ambiguous_names)
    missing_combo_count = len(midterm_missing)

    style_block = """
<style>
body {
  background: radial-gradient(circle at top, #16222d 0%, #0c1218 55%, #090d12 100%) !important;
  color: #dce7ea !important;
}
.jp-Notebook, #notebook-container, .container, div#notebook {
  background: transparent !important;
  color: #dce7ea !important;
}
h1, h2, h3, h4, h5, h6, p, li, strong {
  color: #dce7ea !important;
}
a {
  color: #8fe3cf !important;
}
.hero {
  background: linear-gradient(120deg, #11212b 0%, #1d7874 52%, #f4c95d 100%);
  color: white;
  padding: 28px 32px;
  border-radius: 18px;
  margin-bottom: 20px;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.32);
}
.hero h1, .hero h3, .hero p {
  color: white !important;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
  margin: 18px 0 8px 0;
}
.summary-card {
  background: rgba(15, 24, 31, 0.92);
  border: 1px solid rgba(143, 227, 207, 0.10);
  border-left: 6px solid #1d7874;
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}
.summary-card h4 {
  margin: 0 0 6px 0;
  color: #f5fbfc !important;
}
.summary-card p {
  margin: 0;
  font-size: 0.95rem;
  color: #dce7ea !important;
}
.callout {
  background: rgba(61, 50, 18, 0.70);
  border-left: 6px solid #f4c95d;
  border-radius: 12px;
  padding: 14px 18px;
  margin: 14px 0;
}
.metric-setup {
  background: linear-gradient(90deg, rgba(17, 37, 46, 0.96) 0%, rgba(17, 52, 61, 0.94) 100%);
  border: 1px solid rgba(125, 211, 252, 0.16);
  border-left: 6px solid #38bdf8;
  border-radius: 12px;
  padding: 14px 18px;
  margin: 14px 0 14px 0;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
}
.teacher-takeaway {
  background: linear-gradient(90deg, rgba(7, 45, 35, 0.95) 0%, rgba(10, 68, 47, 0.92) 100%);
  border-left: 6px solid #34d399;
  border-radius: 12px;
  padding: 14px 18px;
  margin: 14px 0;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
}
.insight {
  background: rgba(14, 20, 26, 0.92);
  border: 1px solid rgba(143, 227, 207, 0.12);
  border-left: 6px solid #84a59d;
  border-radius: 12px;
  padding: 14px 18px;
  margin: 14px 0 20px 0;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}
.detail-note {
  background: rgba(9, 14, 18, 0.88);
  border: 1px solid rgba(244, 201, 93, 0.10);
  border-radius: 12px;
  padding: 14px 18px;
  margin: 12px 0 22px 0;
  color: #dce7ea !important;
}
.pipeline-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px;
  margin: 18px 0 8px 0;
}
.pipeline-card {
  background: linear-gradient(180deg, rgba(15, 24, 31, 0.96) 0%, rgba(18, 31, 38, 0.96) 100%);
  border: 1px solid rgba(143, 227, 207, 0.10);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}
.pipeline-step {
  display: inline-block;
  font-size: 0.8rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: #1d7874;
  margin-bottom: 8px;
}
.pipeline-card h4 {
  margin: 0 0 8px 0;
  color: #f5fbfc !important;
}
.pipeline-card p {
  margin: 0;
  font-size: 0.95rem;
  color: #dce7ea !important;
}
.figure-frame {
  background: rgba(12, 18, 24, 0.92);
  border: 1px solid rgba(143, 227, 207, 0.10);
  border-radius: 18px;
  padding: 14px;
  margin: 16px 0 12px 0;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
}
.figure-frame img {
  width: 100%;
  display: block;
  border-radius: 12px;
  background: white;
}
code {
  background: rgba(143, 227, 207, 0.10);
  color: #f4c95d !important;
  padding: 0.14rem 0.32rem;
  border-radius: 6px;
}
</style>
"""

    title_cell = f"""
{style_block}

<div class="hero">
  <h1>Student Engagement with Interactive eBooks and Midterm Success</h1>
  <h3>Stakeholder Report for Course Instruction</h3>
  <p>This report turns Runestone log data into a teacher-facing story about how students used the ebook, which patterns were most closely associated with midterm performance, and what those patterns might mean for course design.</p>
</div>

## Why This Report Exists

The course already uses Runestone heavily for reading checks, lecture practice, Parsons puzzles, coding practice, concept checks, and midterms. The main question behind this report is not just whether students clicked in the ebook, but whether there are meaningful usage patterns that can help an instructor understand who is thriving, who is struggling, and which kinds of practice appear most valuable.
"""

    executive_summary = f"""
## Executive Summary

<div class="summary-grid">
  <div class="summary-card">
    <h4>Midterm 2 Was Lower</h4>
    <p>Across semesters with both exams, midterm 2 averaged <strong>{abs(avg_drop):.1f} points lower</strong> than midterm 1.</p>
  </div>
  <div class="summary-card">
    <h4>Parsons Stood Out</h4>
    <p>Students in the top Parsons first-try quartile scored about <strong>{parsons_gap:.1f} points higher</strong> than the bottom quartile.</p>
  </div>
  <div class="summary-card">
    <h4>Quality Beat Volume</h4>
    <p>Broad, accurate practice was more informative than just logging many clicks or moves.</p>
  </div>
  <div class="summary-card">
    <h4>Effort Alone Was Not Enough</h4>
    <p>High-effort, low-efficiency students still averaged only about <strong>{high_effort_low_eff['mean_exam_score']:.1f}%</strong>.</p>
  </div>
  <div class="summary-card">
    <h4>Practice Midterms Helped</h4>
    <p>Finishing a timed practice exam lined up with about <strong>{practice_finish_gap:.1f} more real-exam points</strong> within the same cohort.</p>
  </div>
</div>

<div class="callout">
<strong>Bottom line:</strong> Nearly all students used the ebook. What separated stronger and weaker performers was not basic participation, but the quality, efficiency, and timing context of that practice.
</div>
"""

    pipeline_cards = "\n".join(
        [
            f"""
<div class="pipeline-card">
  <div class="pipeline-step">Step {int(row.step)}</div>
  <h4>{row.label}</h4>
  <p><strong>{row.count_display}</strong></p>
  <p>{row.description}</p>
</div>
"""
            for row in pipeline_summary.itertuples(index=False)
        ]
    )

    pipeline_section = f"""
## How We Built The Numbers

<div class="pipeline-grid">
{pipeline_cards}
</div>

<div class="callout">
<strong>Why this matters for trust:</strong> the report does not jump straight from raw clicks to conclusions. It first identifies real exam windows, reconstructs student-level midterm scores, then builds pre-exam behavior summaries and compares students within the same semester-midterm cohorts so semester difficulty does not dominate the findings.
</div>
"""

    audience_guide = """
## How To Read This Report

- This report is written for a non-technical audience. Every chart is explained in plain language.
- A **positive relationship** means students with more of that pattern tended to score higher on the midterm.
- A **negative relationship** does not automatically mean the activity was harmful. Sometimes it means the measure is acting like a marker of struggle or late recovery.
- The timed practice-midterm analysis is intentionally conservative. It only uses timed practice sessions that finished before the real exam, and its score uses directly scoreable items like multiple choice, Parsons checks, and unittest-backed coding items.
- The report is organized as a story: first what students did, then what seems to matter most, then what a teacher might do with that information.
"""

    midterm_detection = f"""
## Before The Correlations: How The Midterms Were Identified

Because the raw `problem_name` field was messy and inconsistent, the report first had to solve a naming and labeling problem before any score or engagement analysis could happen.

- The pipeline confidently recovered **{confident_combos} semester-midterm combinations** across the usable semesters.
- Among those confident semesters, there were **{missing_combo_count} missing `(semester, midterm)` combinations**.
- The pipeline also kept **{ambiguous_problem_name_count:,} ambiguous problem-name rows** visible instead of forcing them into the wrong exam.

The key design choice was to trust **timed exam start and finish markers** more than raw names whenever possible. After that, semester labels were standardized and problem names were grouped only when the evidence was strong enough.

<div class="insight">
<strong>Why this matters:</strong> the correlations later in the report only make sense if the exam rows were reconstructed carefully. This step is what turns a messy `problem_name` column into a credible semester-by-midterm structure that can support score reconstruction and comparison.
</div>
"""

    glossary = """
## Plain-Language Glossary

- **Parsons puzzle**: a coding task where students arrange mixed-up code blocks into the right order.
- **Parsons first-try quality**: how often a student’s first graded check on a Parsons problem was already close to correct. In teacher terms, this is a measure of how cleanly a student can set up a solution before a lot of trial-and-error.
- **ActiveCode**: executable coding cells in the ebook. The report uses unit-test results to summarize how well students solved these coding problems.
- **Concept checks**: regular non-exam question interactions in the ebook, often multiple choice.
- **Total events**: every logged action such as a click, move, run, or answer. This is a volume measure.
- **Unique problems**: the number of different problems a student touched. This is a breadth measure.
- **Active days**: the number of different days a student used that activity family before the exam. This is a spacing measure.
"""

    section1 = f"""
## 1. Students Were Already Using The Ebook A Lot

The first thing a teacher needs to know is whether usage was widespread enough for these patterns to matter. The answer is yes.

- Before midterm 1, Parsons usage was already about **{parsons_use_mid1:.1f}%** of students.
- Concept-check usage before midterm 1 was also about **{concept_use_mid1:.1f}%**.
- A typical student had already touched around **{parsons_exposure_mid1:.0f} Parsons problems** and **{activecode_exposure_mid1:.0f} ActiveCode problems** before midterm 1.

This matters because it means the course does not have a simple “some students used the ebook and some did not” story. Most students were already engaging with these tools. That shifts the real question toward **how** they used them.

{metric_setup(f"We started with simple participation because later relationships are hard to interpret if students were not actually exposed to the tool. In this dataset, exposure was already extremely high: Parsons and concept checks were both about {parsons_use_mid1:.1f}% before midterm 1.")}
{md_image("10_participation_rates.png", "Participation rates")}
{detail_note("Who Actually Used Each Activity Type?", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> the course already succeeded in getting students into the ebook ecosystem. That means poor outcomes are probably not being driven by simple non-use. From an instructional perspective, the next layer is not “How do we make students open Runestone?” but “How do we help students use these tools in a way that builds understanding?”
</div>

{metric_setup(f"Once participation looked universal, the next question was depth. The median student had already attempted about {parsons_exposure_mid1:.0f} Parsons problems and {activecode_exposure_mid1:.0f} ActiveCode problems before midterm 1, so we wanted to see whether students were just sampling or doing substantial practice.")}
{md_image("11_typical_exposure.png", "Typical exposure")}
{detail_note("How Much Practice Did A Typical Student Do?", analysis8_notes)}

<div class="insight">
<strong>Student insight:</strong> most students were not just casually sampling the tools. They had already seen many Parsons and ActiveCode problems before the exam. That suggests the ebook was functioning like a real part of the course, not optional enrichment. If students are still underperforming, the likely issue is quality of engagement, confusion on certain topics, or lack of transfer from practice to exam conditions.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> basic participation was already close to universal. The better leverage is not “get students to click once,” but “help students use these tools more effectively and more consistently.”
</div>
"""

    section2 = f"""
## 2. Midterm 2 Was Consistently Harder Than Midterm 1

Across almost every semester, midterm 2 came in below midterm 1. The average drop was about **{abs(avg_drop):.1f} percentage points**, and the largest drop was in **{worst_drop['semester_raw']}**, where midterm 2 was about **{abs(worst_drop['mid2_minus_mid1']):.1f} points lower**.

The score distributions show that this is not just a small pocket of struggling students. In several semesters, the whole score distribution shifts downward for midterm 2.

{metric_setup(f"After confirming students used the ebook heavily, we checked whether exam performance itself followed a stable pattern across terms. The key question here was whether the midterm 2 drop was isolated or repeated across cohorts.")}
{md_image("01_semester_overview.png", "Semester overview")}
{detail_note("Semester Overview", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> when the average score drops across nearly every semester, that points toward something structural rather than accidental. The second exam may be covering more difficult content, asking for more transfer, or arriving at a point in the term when students are under greater time pressure from other courses.
</div>

{metric_setup("Averages can hide whether only a small subgroup struggled, so we looked at the full score distributions next. If the whole distribution shifts lower, that points to a broader course-level challenge rather than just a few outliers.")}
{md_image("02_score_distributions.png", "Score distributions")}
{detail_note("Score Distributions By Semester", analysis8_notes)}

<div class="insight">
<strong>Student insight:</strong> the downward shift is not limited to a small struggling subgroup. A broad distribution shift suggests many students may be feeling the same challenge at once. That is important for teaching decisions, because it argues more for course-wide support before midterm 2 than for a narrow intervention aimed at only a few outliers.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> later material, exam design, pacing, or cumulative fatigue may be making the second exam structurally harder. This is worth treating as a course-level pattern, not a one-off anomaly.
</div>
"""

    section3 = f"""
## 3. The Biggest Pattern: Quality Beat Volume

When we compare many different features of ebook behavior, one theme comes up over and over: **quality measures beat raw activity counts**.

- Parsons first-try quality was the strongest signal in the whole report.
- ActiveCode best score was positive too, but weaker than Parsons.
- Broad concept-check coverage helped, while simple concept-check event volume was slightly negative.

That last point is important. The data suggests that doing well on practice and spreading that practice across more distinct problems is more informative than simply generating lots of events.

{metric_setup("At this point we moved from descriptive questions to predictive ones: which pre-exam behaviors actually traveled with stronger midterm outcomes? The lollipop chart ranks the clearest positive and negative signals in one place.")}
{md_image("03_activity_correlation_lollipop.png", "Strongest activity signals")}
{detail_note("Strongest Activity Signals", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> students do not benefit equally from the same amount of activity. The strongest positive relationships come from doing practice well, while some negative relationships seem to capture confusion, repeated retries, or recovery from a weak starting point. In other words, more activity is not automatically more learning.
</div>

{metric_setup(f"Because one ranked list can make the findings feel fragmented, we also regrouped the results by activity family and metric type. This helps answer whether quality, volume, breadth, or spacing mattered most across different tools.")}
{md_image("04_quality_vs_volume_heatmap.png", "Quality versus volume")}
{detail_note("Quality Versus Volume Heatmap", analysis8_notes)}

<div class="insight">
<strong>Student insight:</strong> this pattern fits a familiar classroom story. A student who touches many distinct questions across multiple days may be building flexible understanding, while a student generating many events on the same material may be signaling uncertainty. The chart suggests that distributed, accurate engagement is healthier than frantic repetition.
</div>

<div class="callout">
<strong>Why might event counts be weak or negative?</strong> A large number of clicks, moves, or checks can reflect productive engagement, but it can also reflect confusion, repeated retries, or revisiting the same material because the student is stuck. Raw volume mixes all of those states together.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> if the goal is to identify readiness, quality indicators are much more useful than simple counts. Broad, accurate, spaced practice looks healthier than intense but messy activity.
</div>
"""

    section4 = f"""
## 4. Parsons Puzzles Were The Clearest Diagnostic Signal

Parsons stood out more than any other activity family. Students in the top quartile of Parsons first-try quality scored about **{parsons_gap:.1f} points higher** than students in the bottom quartile.

This is a strong instructional clue. Parsons puzzles appear to capture something central about success in this course: being able to reason about code structure and put together a correct approach quickly.

{metric_setup(f"Parsons was the strongest overall signal, so we zoomed in. We wanted to see whether that effect looked like a real gradient across students or whether it was being driven by a small corner of the data.")}
{md_image("05_parsons_first_score_scatter.png", "Parsons first-try quality")}
{detail_note("Parsons First-Try Quality Versus Midterm Outcome", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> Parsons appears to be doing more than providing practice. It may be exposing whether students can organize code logic quickly and correctly, which is closely related to exam readiness in an intermediate Python course. That makes it especially useful as a diagnostic, not just an exercise format.
</div>

But the story gets even more interesting when we separate **effort** from **efficiency**. High-effort students were not automatically high-performing. Students who worked a lot on Parsons but still had low first-try quality remained much weaker than students with high quality.

{metric_setup("A strong Parsons relationship could still reflect two different stories: students who practiced more did better, or students who practiced more efficiently did better. So the next chart deliberately separates effort from efficiency.")}
{md_image("13_effort_vs_efficiency.png", "Effort versus efficiency")}
{detail_note("Effort Versus Efficiency", analysis8_notes)}

<div class="insight">
<strong>Student insight:</strong> some students are putting in clear effort but still landing in the low-efficiency region. Those students may be ideal candidates for early intervention because the problem is probably not motivation. It may be conceptual confusion, ineffective study habits, or a need for more scaffolding.
</div>

{metric_setup(f"Scatter plots are useful analytically, but instructors often need something easier to act on. We turned the effort-efficiency map into four simple profiles so the differences in average outcomes become more concrete.")}
{md_image("14_student_profiles.png", "Student profiles")}
{detail_note("Student Behavior Profiles", analysis8_notes)}

<div class="insight">
<strong>What this may mean in the classroom:</strong> these profiles help separate different kinds of need. A low-effort, low-efficiency student may need engagement support, while a high-effort, low-efficiency student may already be trying hard and instead need targeted feedback, tutoring, or conceptual repair.
</div>

The profile view makes this very concrete:

- **{best_profile['profile']}** students averaged about **{best_profile['mean_exam_score']:.1f}%**.
- **{worst_profile['profile']}** students averaged about **{worst_profile['mean_exam_score']:.1f}%**.
- The especially important group is **high effort, low efficiency**. These students were engaged, but still struggled to turn that effort into strong results.

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> Parsons logs may be one of the best early-warning tools in the course. A student with lots of Parsons activity but weak first-try quality is probably not disengaged; they are likely working hard and still need support.
</div>
"""

    section5 = f"""
## 5. Coding Quality Mattered Too, But Not As Strongly

ActiveCode also mattered. Students in the top quartile of ActiveCode best-score quality outscored the bottom quartile by about **{activecode_gap:.1f} points**.

That is meaningful, but it was still smaller than the Parsons effect. This suggests that coding success before the exam matters, but the strongest single early signal may be code-assembly and code-tracing fluency rather than full code-writing performance alone.

{metric_setup("After finding that Parsons was especially strong, we checked whether executable coding practice showed the same pattern. This helps separate whether the strongest signals were specific to Parsons or broader across coding-related activities.")}
{md_image("06_activecode_best_score_scatter.png", "ActiveCode best score")}
{detail_note("ActiveCode Best Practice Quality Versus Midterm Outcome", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> students who can get code-based practice working before the exam tend to arrive more prepared, but the weaker effect compared with Parsons suggests that writing working code is not the only thing that matters. Students may also need strong code-reading, tracing, and structural reasoning skills to succeed on the exam.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> coding practice quality matters, but Parsons may be especially valuable as a fast, sensitive diagnostic of whether students understand program structure well enough to succeed later.
</div>
"""

    section6 = f"""
## 6. Timed Practice Midterms Worked Best As Readiness Checks

This analysis adds a more direct exam-preparation question: when students completed a timed practice midterm before the real exam, did that seem to matter?

The answer is yes, with an important nuance. Merely opening a timed practice exam was not the strongest signal. **Finishing** one was much more informative. Within the same semester-midterm cohort, finishing a timed practice exam was associated with about **{practice_finish_gap:.1f} more real-exam points** than doing no timed practice at all.

- On `mid1`, students who finished a timed practice exam averaged about **{practice_mid1_finished:.1f}%** on the real exam, versus **{practice_mid1_none:.1f}%** for students with no timed practice.
- On `mid2`, the same comparison was about **{practice_mid2_finished:.1f}%** versus **{practice_mid2_none:.1f}%**.

{metric_setup(f"We wanted at least one preparation metric that was more exam-like than ordinary practice counts. Timed practice midterms let us compare students who did no timed practice, students who only opened it, and students who actually completed it under exam-like conditions.")}
{md_image("17_practice_midterm_groups.png", "Timed practice groups")}
{detail_note("Timed Practice Midterms: Participation And Completion", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> completion appears to matter more than simply opening the practice. That fits an intuitive learning story: students who work all the way through exam-like conditions may be rehearsing pacing, retrieving knowledge under pressure, and finding gaps before the real test rather than during it.
</div>

Among students with a finished timed practice exam, **practice quality** mattered too. The conservative timed-practice score used here only includes directly scoreable practice items inside the timed practice window, but even with that restriction the relationship stayed clearly positive. The within-cohort correlation was about **{practice_corr_value:.2f}**, and students in the top timed-practice quality quartile scored about **{practice_quality_gap:.1f} points higher** than students in the bottom quartile.

{metric_setup(f"Completion alone still leaves open the question of whether all finished practice is equally useful. So we also measured the quality of observable work inside the timed practice window and compared that with real-exam performance.")}
{md_image("18_practice_midterm_quality.png", "Timed practice quality")}
{detail_note("Timed Practice Midterms: Quality As A Readiness Check", analysis8_notes)}

<div class="insight">
<strong>Student insight:</strong> stronger performance on the observable parts of a timed practice exam may indicate that a student is not only studying, but studying successfully. For an instructor, this makes practice-midterm quality a potentially useful checkpoint for deciding who may need extra outreach before the actual exam.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> practice midterms seem most valuable as readiness checks when students actually complete them and when instructors pay attention to how well students do on the observable parts. Opening the practice link alone is not the same thing as being prepared.
</div>
"""

    section7 = f"""
## 7. Improvement Was Real, But Starting Point Still Mattered

One of the most important interpretation issues in the report is this: some “negative” relationships are not actually bad news about the activity. They are often a sign that students started from different places.

For example, `Parsons mean score gain` was negative overall (**{parsons_gain_corr:.2f}**). At first glance, that might sound like improvement is associated with worse exam scores. But the more accurate story is:

{improvement_story}

The learning-progression figure shows that both strong and weak exam groups improved over time. The weaker group just started lower and remained lower by exam time.

- Lowest exam quartile on Parsons: from about **{parsons_progression_low['first']:.2f}** to **{parsons_progression_low['last']:.2f}**
- Highest exam quartile on Parsons: from about **{parsons_progression_high['first']:.2f}** to **{parsons_progression_high['last']:.2f}**

{metric_setup("Some of the negative correlations looked counterintuitive, especially score gain. To understand whether improvement was actually helping, we stepped back and compared first observed versus last observed performance for low and high exam groups.")}
{md_image("16_learning_progression.png", "Learning progression")}
{detail_note("Learning Progression", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> weaker students are not failing to learn entirely. Many of them improve substantially. The issue is that they often begin lower and do not fully catch up by exam time. That argues for earlier intervention rather than waiting for exam-week recovery.
</div>

{metric_setup("The progression plot suggested that starting point matters as much as growth, so we made that interpretation explicit in a chart that compares baseline quartiles with later gain and exam outcome.")}
{md_image("12_parsons_baseline_story.png", "Why improvement can mislead")}
{detail_note("Why Improvement Margin Can Mislead", analysis8_notes)}

<div class="insight">
<strong>Student insight:</strong> a big jump in practice score can sometimes be a sign that a student had a lot of ground to make up. In the context of student support, that is still valuable information. It can help identify students who are recovering, but who may still be academically vulnerable.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> improvement margins can be useful, but they should be interpreted alongside where a student started. A large gain may mean a student is recovering from a weak baseline, not that the activity failed.
</div>
"""

    section8 = f"""
## 8. Timing Alone Did Not Explain Success

It is natural to wonder whether stronger students simply used the ebook more consistently and weaker students crammed at the last minute. The data does not support such a clean story.

- Top-performing students still did about **{q4_last7:.1f}%** of their last-28-day activity in the final week before the exam.
- Bottom-performing students did about **{q1_last7:.1f}%**.
- The average number of active days was only slightly different: about **{q4_days:.1f}** for the top quartile versus **{q1_days:.1f}** for the bottom quartile.

That means last-week review seems normal for most students. Timing differences exist, but they are smaller than quality differences.

{metric_setup("A common classroom hypothesis is that weaker students cram while stronger students space their work. We tested that directly by looking at when activity occurred in the 28 days before the exam.")}
{md_image("15_cramming_vs_consistency.png", "Cramming versus consistency")}
{detail_note("Cramming Versus Consistency", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> the data does not support a simplistic “good students never cram” story. Stronger students also study near the exam. A more plausible interpretation is that final-week review is common for everyone, but the students who benefit most are the ones coming in with stronger practice quality and cleaner foundations.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> it is probably not enough to frame the problem as “students should not cram.” Stronger students also ramped up near the exam. The bigger separator was whether that practice was accurate and efficient.
</div>
"""

    section9 = f"""
## 9. Changes Before Midterm 2 Helped Somewhat, Especially Parsons

When we look only at students who had both midterms in the same semester, the effects of changing behavior between midterm 1 and midterm 2 are real, but smaller than the baseline quality effects.

The strongest change-based signal was **{strongest_change['feature'].replace('__change', '').replace('__', ' / ')}**, with a correlation of about **{strongest_change['within_group_rank_corr']:.2f}**.

The clearest practical story was again about Parsons. Students with lower baseline Parsons performance who increased their Parsons activity more before midterm 2 tended to experience a smaller decline from midterm 1 to midterm 2.

{metric_setup("Because midterm 2 drops so consistently, we wanted to know whether post-midterm-1 behavior changes mattered. This chart looks at students with both exams and asks which changes tracked with smaller losses or better recovery.")}
{md_image("07_change_correlation_bars.png", "Change correlations")}
{detail_note("What Changed From Midterm 1 To Midterm 2?", analysis8_notes)}

<div class="insight">
<strong>What this may mean:</strong> behavior change after midterm 1 can help, but it usually does not erase the importance of baseline preparation. This suggests that post-midterm interventions may be most useful as damage control or targeted support, rather than a full substitute for strong early-semester practice.
</div>

{metric_setup("The overall change ranking pointed back to Parsons, so we split that story by starting level. This helps answer whether increased Parsons use was especially relevant for students who began from a weaker position.")}
{md_image("08_parsons_growth_groups.png", "Parsons growth groups")}
{detail_note("Parsons Growth Groups Before Midterm 2", analysis8_notes)}

<div class="insight">
<strong>Student insight:</strong> students who increased Parsons engagement after midterm 1, especially those who were weaker to begin with, tended to soften the drop into midterm 2. That does not prove Parsons caused the improvement, but it does suggest it may be one of the more promising tools to lean on when students are trying to recover.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> behavioral change before the second exam can help, but it seems to work more as a partial recovery tool than as a substitute for stronger foundations earlier in the term.
</div>
"""

    section10 = f"""
## 10. What The Instructor Can Probably Trust Most

Not every item type in the exam logs had a clean, unambiguous correctness signal. Most of the unresolved rows came from fill-in-the-blank style items.

- Mean item-level scoring coverage across the reconstructed exams was still about **{mean_coverage:.1f}%**.
- The largest ambiguous category was fill-in-the-blank, with **{ambiguous_fillb}** unresolved student-problem rows.

This means the strongest conclusions in the report are being driven by the cleaner parts of the data: Parsons, ActiveCode, and multiple-choice style signals.

{metric_setup(f"A final report should also show where the data is less certain. We therefore counted which exam item families were left unscored because the logs were ambiguous, rather than pretending every row was equally trustworthy.")}
{md_image("09_scoring_limitations.png", "Scoring limitations")}
{detail_note("Scoring Limitations", analysis8_notes)}

<div class="insight">
<strong>Why this matters:</strong> a trustworthy report should say where it is strong and where it is weaker. By separating ambiguous item types instead of forcing questionable scores, the analysis protects the main conclusions from being inflated by noisy data. That makes the cleaner findings more credible.
</div>

<div class="teacher-takeaway">
<strong>Teacher takeaway:</strong> the report is most trustworthy when it speaks about Parsons, coding practice, broad concept-check usage, and exam-level trends. It is less definitive for noisier item types like some fill-in-the-blank questions.
</div>
"""

    recommendations = """
## Practical Recommendations For The Course Team

Based on the full story in the report, here are the most actionable ideas for instruction:

1. **Use Parsons as an early diagnostic tool.** Students with repeated low first-check quality look like an important support group.
2. **Watch for high-effort, low-efficiency students.** These students appear engaged, but they are not converting activity into understanding.
3. **Promote broad and accurate practice, not just more clicks.** Breadth and quality were more useful than raw event volume.
4. **Encourage students to finish timed practice midterms.** Completion looked more informative than simply opening a practice exam.
5. **Do not overinterpret last-week studying.** Stronger students also did a lot of work close to the exam; timing alone is not the main story.
6. **Look closely at the midterm 2 pipeline.** The across-semester drop suggests a structural challenge in the later part of the course.
7. **Interpret improvement in context.** Large gains can indicate recovery from a weak starting point rather than a fully solved problem.
"""

    closing = """
## Closing Thought

The clearest overall lesson from this report is that the ebook is already an important part of the course. The opportunity is not simply to increase usage, but to use the ebook data more strategically:

- to identify students who are putting in effort but still struggling,
- to notice when students are practicing broadly and cleanly,
- and to intervene earlier, especially before the second midterm.

That is where the data appears most actionable for teaching.
"""

    coverage_section = """
## What This HTML Covers

This report intentionally includes the full chain of analysis so it can stand on its own as the final deliverable:

1. **Midterm identification and semester normalization** from messy `problem_name` values.
2. **Timed exam window reconstruction** using start and finish markers.
3. **Student-level midterm score reconstruction** from observable item correctness.
4. **Participation and exposure summaries** by activity family.
5. **Semester-level score patterns** including the persistent midterm 2 drop.
6. **Activity-type importance analysis** across Parsons, ActiveCode, concept checks, multiple choice, and other families.
7. **Effort versus efficiency and student behavior profiles**.
8. **Timed practice-midterm participation and quality analysis**.
9. **Learning progression and improvement-margin interpretation**.
10. **Cramming versus consistency timing analysis**.
11. **Mid1-to-mid2 behavior-change analysis**.
12. **Scoring limitations, ambiguity handling, and confidence notes**.

The goal was to make the HTML useful not just as a slide deck, but as a standalone analytical report that explains both the findings and how those findings were produced.
"""

    appendix = """
## Appendix: Files Behind This Report

This report was built from the saved outputs in `analysis8_outputs`, especially:

- `student_midterm_scores.parquet`
- `student_midterm_activity_features.parquet`
- `activity_correlations.csv`
- `activity_change_correlations.csv`
- `participation_summary.csv`
- `typical_exposure_summary.csv`
- `student_profile_summary.csv`
- `learning_progression_summary.csv`
- `cramming_bin_summary.csv`
- `cramming_student_metrics.csv`
- `pipeline_summary.csv`
- `practice_midterm_group_summary.csv`
- `practice_midterm_correlation_summary.csv`

For a live interactive version of the same story, use the Streamlit app in `stakeholder_report_app.py`.
"""

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(title_cell),
        nbf.v4.new_markdown_cell(executive_summary),
        nbf.v4.new_markdown_cell(pipeline_section),
        nbf.v4.new_markdown_cell(midterm_detection),
        nbf.v4.new_markdown_cell(audience_guide),
        nbf.v4.new_markdown_cell(glossary),
        nbf.v4.new_markdown_cell(section1),
        nbf.v4.new_markdown_cell(section2),
        nbf.v4.new_markdown_cell(section3),
        nbf.v4.new_markdown_cell(section4),
        nbf.v4.new_markdown_cell(section5),
        nbf.v4.new_markdown_cell(section6),
        nbf.v4.new_markdown_cell(section7),
        nbf.v4.new_markdown_cell(section8),
        nbf.v4.new_markdown_cell(section9),
        nbf.v4.new_markdown_cell(section10),
        nbf.v4.new_markdown_cell(recommendations),
        nbf.v4.new_markdown_cell(closing),
        nbf.v4.new_markdown_cell(coverage_section),
        nbf.v4.new_markdown_cell(appendix),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3"}
    with (BASE_DIR / "analysis9.ipynb").open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)
    export_standalone_html(nb.cells)


if __name__ == "__main__":
    build_report()
