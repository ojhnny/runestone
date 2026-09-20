"""Create the walkthrough notebooks. Run from the repo root if you edit the cells."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    path = NOTEBOOKS / name
    path.write_text(nbf.writes(nb), encoding="utf-8")
    print("wrote", path)


write(
    "01_explore_event_log_with_sql.ipynb",
    [
        md(
            """
# DuckDB walkthrough: 20M Runestone events

This notebook is the "what's in the log?" pass. I'm using DuckDB because the
file is ~20 million rows and I don't want to pull all of it into pandas just
to count things.

You need `runestone_event_log.parquet` in the project root. If a cell errors on that,
the rest of the analysis can still use the saved CSVs from `python -m ebook_analysis`.
"""
        ),
        code(
            """
from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from ebook_analysis.duckdb_queries import connect, query, run_pre_exam_views

con = connect()
con.execute("SELECT 1").fetchone()
"""
        ),
        md(
            """
First, the obvious checks: how many rows, how many semesters, how far back the log goes.
This is the same query as `sql/01_event_overview.sql`.
"""
        ),
        code(
            """
overview = query(con, "01_event_overview.sql")
overview.T
"""
        ),
        md(
            """
Semester counts are a good "did all the files actually load?" check. F21 through F25
plus the winter terms should show up. If one bar is tiny, that semester's export
probably didn't make it into `runestone_event_log.parquet`.
"""
        ),
        code(
            """
by_semester = query(con, "02_events_by_semester.sql")
by_semester
"""
        ),
        md(
            """
Exam windows are not inferred here. `exam_windows/` already has timed start/finish
markers. We just collapse extra sessions into one window per student and exam.
"""
        ),
        code(
            """
windows = query(con, "03_exam_windows.sql")
windows.groupby(["semester_raw", "midterm"]).size().rename("n_windows").reset_index().head(20)
"""
        ),
        md(
            """
`04_pre_exam_events.sql` builds a view of practice *before* each student's exam.
That's the leaky part to get right: if exam clicks sneak into "practice," every
quality feature looks better than it is.
"""
        ),
        code(
            """
pre_count = run_pre_exam_views(con)
pre_count
"""
        ),
        code(
            """
# peek at a few pre-exam rows without dumping the whole view
con.execute('''
    SELECT semester_raw, midterm, family, days_before, level_chapter, problem_name
    FROM pre_exam_events
    WHERE family = 'parsons'
    LIMIT 8
''').df()
"""
        ),
        md(
            """
That's enough to trust the SQL layer. Notebook 02 joins these features onto
midterm scores and looks at Parsons.
"""
        ),
    ],
)

write(
    "02_practice_features_and_parsons.ipynb",
    [
        md(
            """
# Features and the Parsons gap

The original analysis already had quality features (first / best score by
activity family). The DuckDB layer adds coverage, timing, and mix. This notebook
reads the merged table so we aren't rescanning 20M rows.
"""
        ),
        code(
            """
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(ROOT))

from ebook_analysis.stats import add_cohort, compute_corr_table, within_group_quartile_gap

out = ROOT / "analysis_outputs"
features = pd.read_parquet(out / "student_exam_features_sql.parquet")
corr = pd.read_csv(out / "activity_correlations_with_sql.csv")
features.shape
"""
        ),
        md(
            """
Quick look at the new columns. Anything starting with `coverage__`, `timing__`,
`consistency__`, or `mix__` came from SQL.
"""
        ),
        code(
            """
new_cols = [c for c in features.columns if c.startswith(("coverage__", "timing__", "consistency__", "mix__"))]
new_cols
features[new_cols].describe().T.head(20)
"""
        ),
        md(
            """
The number everyone quotes is the within-cohort quartile gap for Parsons
first-try quality. "Within cohort" just means: rank students against other
people taking the same midterm in the same semester. Otherwise F21 mid1 and
W24 mid2 get dumped into one comparison, which is messy.
"""
        ),
        code(
            """
parsons = corr.loc[corr["feature"].eq("parsons__mean_first_score")].iloc[0]
parsons
"""
        ),
        code(
            """
gap = within_group_quartile_gap(
    add_cohort(features), "cohort", "parsons__mean_first_score", "score_pct"
)
print(f"top vs bottom Parsons quartile: {gap:.1f} points")
"""
        ),
        md(
            """
Do the new coverage / timing features even matter? Quality still wins, but
chapter coverage is a nicer "did they actually touch the topics?" check than
raw event counts.
"""
        ),
        code(
            """
focus = [
    "parsons__mean_first_score",
    "mchoice__mean_first_score",
    "coverage__unique_chapters",
    "coverage__chapter_diversity",
    "timing__share_last_7d",
    "consistency__span_days",
    "mix__coding_share",
    "all_activity__total_events",
]
corr.loc[corr["feature"].isin(focus)].sort_values("within_group_rank_corr", ascending=False)
"""
        ),
        md(
            """
One thing that looks backwards until you sit with it: `mix__parsons_share` is
*negatively* related to exam score, while Parsons first-try quality is the
strongest positive feature. Students who spend a huge fraction of their
clicks on Parsons but don't get them right on the first check are exactly
the high-effort / low-efficiency group. Volume without quality is not a
good sign.
"""
        ),
        code(
            """
corr.loc[corr["feature"].isin(["parsons__mean_first_score", "mix__parsons_share", "mix__activecode_share"])]
"""
        ),
        code(
            """
quartiles = pd.read_csv(out / "feature_quartile_scores.csv")
quartiles.loc[quartiles["feature"].eq("parsons__mean_first_score")].groupby("feature_quartile", observed=False)["mean_score_pct"].mean()
"""
        ),
    ],
)

write(
    "03_cohort_models_and_at_risk.ipynb",
    [
        md(
            """
# Cohort models and a small at-risk baseline

The 17.5-point gap is easy to show in a meeting. This notebook is the
follow-up I'd want in an interview:

- Does the Parsons relationship survive semester x exam fixed effects?
- What happens if we also control for how much they practiced?
- If we try to flag bottom-quartile students before the exam, how well does
  a simple model do when we hold out a whole semester?
"""
        ),
        code(
            """
from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(ROOT))

out = ROOT / "analysis_outputs"
regression = pd.read_csv(out / "parsons_regression.csv")
at_risk = pd.read_csv(out / "at_risk_cv.csv")
coefs = pd.read_csv(out / "at_risk_coefs.csv")
headlines = pd.read_csv(out / "headline_metrics.csv")
"""
        ),
        md(
            """
`parsons_first` is on a 0-100 scale here, so the coefficient is "exam points
associated with a 1-point increase in Parsons first-try percent." The CI is
the part to look at, not just the point estimate.
"""
        ),
        code(
            """
regression
"""
        ),
        md(
            """
If the effort-controlled spec is still positive, we can at least say this
isn't *only* "people who click more score more." It can still be ability.
Don't say causal.
"""
        ),
        code(
            """
headlines
"""
        ),
        md(
            """
The at-risk model is logistic regression, median-imputed, standardized,
`GroupKFold` by semester. I wouldn't ship this as a product. It's there to
show the features have some out-of-semester signal.
"""
        ),
        code(
            """
at_risk
"""
        ),
        code(
            """
coefs.sort_values("coef")
"""
        ),
        md(
            """
If Parsons first-try has a negative coefficient here, that's the expected
direction: higher first-try quality, lower chance of a bottom-quartile exam.
Volume features can flip around once quality is in the model.
"""
        ),
    ],
)
