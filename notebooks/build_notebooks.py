"""Generate the build notebooks (01-07) from this file.

The notebooks are the readable record of how every result was produced: each
one runs one pipeline stage step by step, prints the source of each function
or SQL file just before calling it, and shows aggregate results in between.
The logic itself lives in ``src/so_ell`` (tested); the notebooks call it in
the same order as ``so_ell.pipeline`` and write the same files.

    uv run python notebooks/build_notebooks.py   # regenerate the .ipynb files
    make notebooks                               # execute them in order (needs data/raw)

Executed notebooks only ever display counts, means and other aggregates, never
student-level rows, so they are safe to commit for a stakeholder project.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = """
import pandas as pd
from so_ell import config
from so_ell.notebook import show, show_sql, describe
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
config.ensure_dirs()
T = config.TABLES_DIR
"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    (HERE / name).write_text(nbf.writes(nb), encoding="utf-8")
    print("wrote", name)


# --------------------------------------------------------------------------- 01
write(
    "01_ingest.ipynb",
    [
        md("""
# 01 · Ingest: raw Datashop exports → one event log

**Input:** 13 compressed Runestone Datashop exports in `data/raw/<semester>/` (not in git; see `data/raw/README.md`).
**Output:** `data/interim/events.parquet`, one row per logged ebook event, plus row-count and data-quality tables.

What this stage does, in order:
1. Verify each raw file against `data/raw/MANIFEST.csv` (sha256), so results are tied to an exact set of inputs.
2. Read each export as text, normalize the column names (the exports drift between semesters).
3. **Namespace student and session ids by semester and section.** Each section's export numbers students from 1, so a bare id is not unique.
4. Write one sorted parquet with DuckDB, then run SQL checks on it.
"""),
        code(SETUP),
        md("## 1. The manifest: which files, which semester, which checksum"),
        code("""
from so_ell import ingest
show(ingest.infer_section, ingest.build_manifest)
manifest = pd.read_csv(config.RAW_MANIFEST)
manifest[["semester", "section", "path", "bytes"]]
"""),
        md("## 2. Reading one export\nEverything is read as text first (`dtype=str`) so nothing is silently coerced; ids are rebuilt as `F25-MW-0005`."),
        code("show(ingest.standardize_columns, ingest.read_export)"),
        md("## 3. Run the ingest\nChecksums are verified, each file is written as a parquet part, and DuckDB concatenates and sorts them by semester, student and time."),
        code("""
show(ingest.run_ingest)
counts = ingest.run_ingest(verify_checksums=True)
counts.to_csv(config.QUALITY_DIR / "ingest_row_counts.csv", index=False)
counts
"""),
        code("""print(f"{counts['rows'].sum():,} events in the combined log")"""),
        md("W20 and W21 are near-empty exports (404 and 5 rows, no timed exams), which is why the analysis covers 8 semesters: "),
        code("""print("analysis semesters:", config.ANALYSIS_SEMESTERS, "| excluded:", config.EXCLUDED_SEMESTERS)"""),
        md("## 4. Data-quality checks on the event log\n*Hard* checks fail the pipeline; *soft* checks are reported."),
        code("""
from so_ell import validation
show(validation.event_log_checks)
checks = validation.event_log_checks()
checks.to_csv(config.QUALITY_DIR / "event_log_checks.csv", index=False)
checks
"""),
        md("""
## 5. Why the ids had to be namespaced

How many local ids appear in **both** section exports of a semester and started the real midterm in both?
In the original project these were merged into one "student".
"""),
        code("""
import duckdb
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
EV = f"read_parquet('{config.EVENTS_PARQUET.as_posix()}')"
con.execute(f'''
WITH starts AS (
    SELECT semester, section, local_student_id
    FROM {EV}
    WHERE selection = 'timedExam' AND action = 'start'
      AND regexp_matches(lower(problem_name), 'mid')
      AND NOT regexp_matches(lower(problem_name), 'practice')
    GROUP BY ALL
)
SELECT semester, COUNT(*) AS local_ids_in_both_sections
FROM (SELECT semester, local_student_id FROM starts GROUP BY ALL HAVING COUNT(DISTINCT section) = 2)
GROUP BY semester ORDER BY semester
''').df()
"""),
    ],
)

# --------------------------------------------------------------------------- 02
write(
    "02_exam_windows.ipynb",
    [
        md("""
# 02 · Exam windows: when did each student take each real midterm?

Runestone logs a `timedExam` *start* and *finish* marker for every timed assessment. The same mechanism is
used for practice exams and in-class quizzes, so the first job is deciding which names are the real midterms.
The window `[start, finish]` is used twice later: to score the exam, and to keep exam answers out of "practice".
"""),
        code(SETUP),
        md("## 1. Which timed exams count as a real midterm"),
        code("""
from so_ell import exam_windows as ew
show(ew.parse_midterm, ew.is_real_exam_name)
print("excluded words:", ew.NOT_REAL_EXAM_PATTERN)
"""),
        md("## 2. Load the start/finish markers (DuckDB, reading only the rows needed)"),
        code("""
show(ew.load_timed_markers)
markers = ew.load_timed_markers()
print(f"{len(markers):,} timedExam markers")
markers.groupby(["semester_raw", "action"]).size().unstack()
"""),
        md("## 3. Every timed-exam name, and whether it was kept\nThis audit table is the evidence for the name rule above."),
        code("""
show(ew.anchor_name_audit)
audit = ew.anchor_name_audit(markers)
audit.to_csv(T / "exam_name_audit.csv", index=False)
audit.sort_values("students", ascending=False).head(30)
"""),
        md("## 4. Build one attempt per student, session and midterm"),
        code("""
show(ew.build_attempts, ew.run_exam_windows)
result = ew.run_exam_windows()
attempts = result["attempts"]
from so_ell import validation
validation.validate(attempts, validation.exam_attempts_schema)
assert result["missing"].empty, result["missing"]
result["summary"].to_csv(T / "exam_window_summary.csv", index=False)
print(f"{len(attempts):,} attempts; every semester has both midterms")
result["summary"]
"""),
        code("""
attempts.groupby("midterm")["attempt_minutes"].describe().round(1)
"""),
        md("""
## 5. Why the windows matter for practice features

Count answer events that fall inside a student's **midterm 1** window but before their **midterm 2**.
A naive `time < exam_start` filter would count every one of these as "practice for midterm 2".
"""),
        code("""
import duckdb
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
con.register("attempts", attempts)
EV = f"read_parquet('{config.EVENTS_PARQUET.as_posix()}')"
con.execute(f'''
WITH w AS (SELECT semester_raw, anon_student_id, midterm, MIN(attempt_start) s, MAX(attempt_end) e FROM attempts GROUP BY ALL)
SELECT COUNT(*) AS mid1_exam_events_before_mid2
FROM {EV} ev
JOIN w m1 ON m1.semester_raw = ev.semester AND m1.anon_student_id = ev.anon_student_id AND m1.midterm = 'mid1'
JOIN w m2 ON m2.semester_raw = ev.semester AND m2.anon_student_id = ev.anon_student_id AND m2.midterm = 'mid2'
WHERE ev.time BETWEEN m1.s AND m1.e AND ev.time < m2.s
''').df()
"""),
    ],
)

# --------------------------------------------------------------------------- 03
write(
    "03_scoring.ipynb",
    [
        md("""
# 03 · Reconstructing midterm scores from the log

The gradebook isn't in the data, so each student's midterm score is rebuilt from the answers they logged
inside their exam window. Rules per question type:

| type | rule |
|---|---|
| multiple choice | last answer submitted, correct/incorrect as logged |
| Parsons | best attempt; opened but never submitted = 0 |
| ActiveCode | best unit-test pass rate; never run = 0 |
| fill-in / clickable / drag-and-drop | compared to an **inferred** answer key (see below) |
| short answer | not scoreable, excluded and reported |
"""),
        code(SETUP),
        md("## 1. Parsing the logged answers"),
        code("""
from so_ell import scoring, scores
show(scoring.activity_family, scoring.parse_mchoice_score, scoring.parse_parsons_score, scoring.parse_unittest_pct, scoring.normalize_response)
"""),
        md("## 2. Load only the answer events\nPage views and navigation are skipped; only answer selections are read, and the large text columns are kept only where a parser needs them."),
        code("""
show(scores.load_scoreable_events)
raw = scores.load_scoreable_events()
print(f"{len(raw):,} answer events; {raw.memory_usage(deep=True).sum() / 1e9:.1f} GB in memory")
raw["selection"].value_counts().to_frame("events")
"""),
        md("## 3. Exam windows and per-event scores"),
        code("""
show(scores.build_windows, scores.add_event_scores, scores.label_exam_events)
windows = scores.build_windows(pd.read_parquet(config.EXAM_ATTEMPTS_PARQUET))
print(f"{len(windows):,} student-exam windows")
"""),
        md("""
## 4. Inferred answer keys

Fill-in, clickable and drag-and-drop events don't carry a correct/incorrect flag. For each problem the key is the
most common answer among students who reached a final answer, used only when at least 65% agree
(`KEY_CONFIDENCE_THRESHOLD`). Notebook 05 re-runs the headline result at thresholds from 55% to "no inferred keys".
"""),
        code("""
show(scores.infer_answer_keys)
print("threshold:", config.KEY_CONFIDENCE_THRESHOLD)
"""),
        md("## 5. Score every student-exam"),
        code("""
show(scores._score_problem, scores.score_exams)
result = scores.score_exams(raw, windows)
from so_ell import validation
validation.validate(result["student_scores"], validation.student_scores_schema)
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
result["problem_scores"].to_parquet(config.PROCESSED_DIR / "student_exam_problem_scores.parquet", index=False)
result["student_scores"].to_parquet(config.PROCESSED_DIR / "student_exam_scores.parquet", index=False)
result["semester_summary"].to_csv(T / "cohort_score_summary.csv", index=False)
result["answer_keys"].drop(columns=["answer_key"]).to_csv(T / "inferred_answer_keys.csv", index=False)
result["unsupported_summary"].to_csv(T / "unscored_items_summary.csv", index=False)
print(f"{len(result['student_scores']):,} student-exam scores, "
      f"{result['student_scores']['anon_student_id'].nunique():,} students")
"""),
        md("## 6. Results by cohort\nExam difficulty moves a lot between semesters, which is why every later comparison is made *within* a semester × midterm cohort."),
        code("""result["semester_summary"].round(2)"""),
        md("Answer-key confidence (keys themselves are not exported):"),
        code("""
keys = result["answer_keys"]
pd.Series({
    "problems with a candidate key": len(keys),
    "keys used (>= threshold)": int((keys["key_confidence"] >= config.KEY_CONFIDENCE_THRESHOLD).sum()),
    "median agreement": round(float(keys["key_confidence"].median()), 3),
}).to_frame("value")
"""),
        md("Items that could not be scored:"),
        code("""result["unsupported_summary"]"""),
        code("del raw"),
    ],
)

# --------------------------------------------------------------------------- 04
write(
    "04_features.ipynb",
    [
        md("""
# 04 · Pre-exam practice features

For every student and midterm, summarize the practice they did **before** that exam. Two leakage guards:

1. Every event inside **any** of the student's exam windows is removed first (so midterm 1 answers are never midterm 2 practice).
2. Only events before `exam_start - cutoff_days` are used. `cutoff_days = 0` is the main feature set; 3, 7, 14, 21 and 28 feed the early-warning curve in notebook 06.

Two engines build complementary features: pandas (per-problem quality: first try, best, gain) and DuckDB SQL
(coverage, timing, consistency). Both apply the same guards; the SQL version is shown in full below.
"""),
        code(SETUP),
        md("## 1. Practice events with exam windows removed"),
        code("""
from so_ell import features as F, scores, sql_features, validation
from so_ell.config import WINDOW_KEYS
show(F.drop_exam_window_events, F.score_study_events)
windows = scores.build_windows(pd.read_parquet(config.EXAM_ATTEMPTS_PARQUET))
student_scores = pd.read_parquet(config.PROCESSED_DIR / "student_exam_scores.parquet")
raw = scores.load_scoreable_events()
study = F.score_study_events(raw, windows)
print(f"{len(raw):,} answer events -> {len(study):,} outside every exam window")
del raw
"""),
        md("## 2. Pandas features: per-problem quality"),
        code("""
show(F.summarize_problems, F.build_practice_features)
practice = F.build_practice_features(study, windows, student_scores, cutoff_days=0)
practice.shape
"""),
        md("## 3. SQL features (DuckDB)\nThe event log is queried in place; these files are executed in order."),
        code("""
show(sql_features.connect, sql_features.build_sql_features)
show_sql("04_pre_exam_events.sql", "05_student_exam_features.sql", "06_chapter_diversity.sql", "08_timing_bins.sql")
sql = sql_features.build_sql_features(cutoff_days=0)
sql["student_features"].shape
"""),
        md("## 4. Merge, validate, save"),
        code("""
from so_ell import pipeline
extra = [c for c in sql["student_features"].columns if c not in WINDOW_KEYS]
features = practice.merge(sql["student_features"][WINDOW_KEYS + extra], on=WINDOW_KEYS, how="left")
validation.validate(features, validation.features_schema)
features.to_parquet(config.PROCESSED_DIR / "student_exam_features.parquet", index=False)
sql["overview"].to_csv(T / "event_log_overview.csv", index=False)
sql["by_semester"].to_csv(T / "events_by_semester.csv", index=False)
sql["chapter_cohort"].to_csv(T / "chapter_activity_by_cohort.csv", index=False)
pipeline.public_timing_table(sql["timing_bins"], features).to_csv(T / "practice_timing_by_score_band.csv", index=False)
print(f"{features.shape[0]:,} student-exams x {features.shape[1]} columns")
"""),
        md("Feature families (column prefix before `__`):"),
        code("""
fam = pd.Series([c.split("__")[0] for c in features.columns if "__" in c]).value_counts()
fam.to_frame("columns")
"""),
        md("Summary of the main features (aggregate only). The full list with definitions is in `docs/feature_dictionary.md`."),
        code("""
from so_ell.models import PRACTICE_FEATURES
describe(features, [c for c in PRACTICE_FEATURES if c in features])
"""),
        md("## 5. Leakage check\nNo practice event may sit inside any exam window. Count events in `study` that fall inside one of the student's windows (should be 0), versus the same count on the unfiltered answer events. The synthetic end-to-end test (`tests/test_pipeline_end_to_end.py`) asserts the same thing for both feature engines."),
        code("""
def events_inside_windows(events):
    probe = events[["semester_raw", "anon_student_id", "time"]].merge(
        windows[["semester_raw", "anon_student_id", "exam_start", "exam_end"]], on=["semester_raw", "anon_student_id"]
    )
    return int(probe["time"].between(probe["exam_start"], probe["exam_end"]).sum())

print("answer events inside an exam window, before the guard:", f"{events_inside_windows(scores.load_scoreable_events()):,}")
print("practice events inside an exam window, after the guard:", f"{events_inside_windows(study):,}")
"""),
        md("## 6. Early-warning feature sets\nThe same code with the line moved N days before the exam."),
        code("""
for cutoff in [c for c in pipeline.LEAD_TIME_CUTOFFS if c]:
    early = F.build_practice_features(study, windows, student_scores, cutoff_days=cutoff)
    early_sql = sql_features.build_sql_features(cutoff_days=cutoff)["student_features"]
    early = early.merge(early_sql[WINDOW_KEYS + [c for c in early_sql.columns if c not in WINDOW_KEYS]], on=WINDOW_KEYS, how="left")
    early.to_parquet(config.PROCESSED_DIR / f"student_exam_features_cutoff_{cutoff}d.parquet", index=False)
    print(f"cutoff {cutoff:>2}d: mean events per student-exam {early['all_activity__total_events'].mean():7.1f}")
"""),
        code("del study"),
    ],
)

# --------------------------------------------------------------------------- 05
write(
    "05_analysis.ipynb",
    [
        md("""
# 05 · What kind of practice lines up with midterm scores?

All comparisons are made **within cohort** (semester × midterm), because exam difficulty varies a lot between terms.
1. Screen every numeric feature (116) with within-cohort rank correlations, controlling the false discovery rate.
2. Quantify the headline (Parsons first-try success) with a quartile gap, a student-clustered bootstrap, and fixed-effects regressions.
3. Check it doesn't hinge on the inferred answer keys.

These are **associations**: students who solve Parsons problems on the first try may simply be stronger students.
The midterm-2-controlling-for-midterm-1 model is the closest this data gets to separating the two.
"""),
        code(SETUP),
        code("""
from so_ell import stats, pipeline
from so_ell.features import mid1_to_mid2_changes
features = stats.add_cohort(pd.read_parquet(config.PROCESSED_DIR / "student_exam_features.parquet"))
print(f"{len(features):,} student-exams in {features['cohort'].nunique()} cohorts")
"""),
        md("## 1. Screening every feature"),
        code("""
show(stats.within_group_rank_corr, stats.within_group_quartile_gap, stats.benjamini_hochberg, stats.compute_corr_table)
corr = stats.compute_corr_table(features, "cohort", "score_pct", pipeline.numeric_feature_cols(features))
corr.to_csv(T / "feature_signals.csv", index=False)
print(f"{(corr['q_value_bh'] < 0.05).sum()} of {len(corr)} features significant at q < 0.05")
corr.head(15)[["feature", "n", "within_group_rank_corr", "top_quartile_minus_bottom_quartile_pct_points", "q_value_bh"]].round(4)
"""),
        md("Volume of activity is a weak signal, and raw event counts are even negative (more repeated wrong checks), compared with the *quality* of first attempts:"),
        code("""corr[corr.feature.str.contains("total_events|active_days")][["feature", "within_group_rank_corr", "q_value_bh"]].round(4)"""),
        md("## 2. How big is the Parsons gap, and how sure are we?\nStudents (not rows) are resampled, because one student contributes two midterms."),
        code("""
show(stats.bootstrap_quartile_gap)
boot = stats.bootstrap_quartile_gap(features, n_boot=1000)
pd.DataFrame([boot]).to_csv(T / "parsons_gap_bootstrap.csv", index=False)
pd.DataFrame([boot]).round(2)
"""),
        md("## 3. Fixed-effects regressions\nScore on Parsons first-try %, with cohort fixed effects and student-clustered standard errors. Spec 3 asks: among students with the *same* midterm 1 score, does Parsons practice before midterm 2 still line up with midterm 2?"),
        code("""
show(stats._prep_model_frame, stats._fit_clustered, stats.fit_parsons_models)
fe = stats.fit_parsons_models(features)
fe.to_csv(T / "parsons_models.csv", index=False)
fe[["spec", "n", "coef", "std_err", "ci_low", "ci_high", "p_value", "r_squared"]].round(4)
"""),
        md("Coefficients are exam points per Parsons point; ×10 gives points per 10 Parsons points."),
        md("## 4. Quartile tables for the dashboard"),
        code("""
show(stats.quartile_table)
quartiles = pd.concat([stats.quartile_table(features, c) for c in pipeline.HEADLINE_FEATURES if c in features], ignore_index=True)
quartiles.to_csv(T / "feature_quartile_scores.csv", index=False)
quartiles.loc[quartiles["feature"].eq(stats.PARSONS)].groupby("feature_quartile")[["mean_score_pct"]].mean().round(1)
"""),
        md("## 5. Change between midterms\nDid students who changed their practice between midterm 1 and 2 change their score?"),
        code("""
show(mid1_to_mid2_changes, pipeline.parsons_change_groups)
change = mid1_to_mid2_changes(features)
change_cols = [c for c in change.columns if c.endswith("__change")]
stats.compute_corr_table(change, "semester_raw", "score_change", change_cols).to_csv(T / "change_signals.csv", index=False)
groups = pipeline.parsons_change_groups(change)
groups.to_csv(T / "parsons_change_groups.csv", index=False)
groups.round(2)
"""),
        md("## 6. Sensitivity to the inferred answer keys\nRe-score every exam at each key threshold (1.01 = drop inferred keys entirely) and refit."),
        code("""
show(pipeline.key_threshold_sensitivity)
sens = pipeline.key_threshold_sensitivity(features)
sens.to_csv(T / "key_threshold_sensitivity.csv", index=False)
sens.round(3)
"""),
    ],
)

# --------------------------------------------------------------------------- 06
write(
    "06_models.ipynb",
    [
        md("""
# 06 · Flagging students before the exam

**Target:** bottom quartile of the student's own cohort (so every semester contributes its weakest 25%).
**Validation:** leave one semester out: train on 7 semesters, predict the unseen 8th, as an instructor would use it.
**Models:** regularized logistic regression (interpretable baseline) vs histogram gradient boosting.
Every run is logged to MLflow (`make mlflow` opens the UI).
"""),
        code(SETUP),
        code("""
from so_ell import models, pipeline
features = pd.read_parquet(config.PROCESSED_DIR / "student_exam_features.parquet")
print("features used:", models.PRACTICE_FEATURES)
"""),
        md("## 1. Model definitions and cross-validation"),
        code("""
show(models.make_model, models.add_target, models.precision_at_top, models.leave_one_semester_out)
"""),
        md("## 2. Run the suite\nThe last three rows are midterm 2 only: does practice add anything once midterm 1 is known?"),
        code("""
show(models.run_model_suite)
results = models.run_model_suite(features)
comparison = pd.DataFrame([{"model": k} | r.summary for k, r in results.items()])
comparison.to_csv(T / "at_risk_model_comparison.csv", index=False)
pd.concat([r.folds for r in results.values()], ignore_index=True).to_csv(T / "at_risk_folds.csv", index=False)
results["gradient_boosting"].importance.to_csv(T / "at_risk_importance_gbm.csv", index=False)
results["logistic"].importance.to_csv(T / "at_risk_coefficients_logistic.csv", index=False)
comparison.round(3)
"""),
        md("Per held-out semester (AUC):"),
        code("""pd.concat([r.folds for r in results.values()]).pivot(index="held_out_semester", columns="model", values="auc").round(3)"""),
        md("## 3. What the models rely on"),
        code("""results["gradient_boosting"].importance.round(4)"""),
        code("""results["logistic"].importance.round(3)"""),
        md("## 4. Calibration (out of fold)"),
        code("""
show(models.calibration_table)
oof = pd.concat([results["logistic"].oof, results["gradient_boosting"].oof], ignore_index=True)
cal = models.calibration_table(oof)
cal.to_csv(T / "at_risk_calibration.csv", index=False)
cal.round(3)
"""),
        md("## 5. How early does the signal appear?\nThe same models, trained on features from practice at least N days before the exam (built in notebook 04)."),
        code("""
show(models.lead_time_curve)
by_cutoff = {0: features}
for c in [c for c in pipeline.LEAD_TIME_CUTOFFS if c]:
    by_cutoff[c] = pd.read_parquet(config.PROCESSED_DIR / f"student_exam_features_cutoff_{c}d.parquet")
lead = models.lead_time_curve(by_cutoff)
lead.to_csv(T / "at_risk_lead_time.csv", index=False)
lead.pivot(index="cutoff_days", columns="model", values="auc").sort_index(ascending=False).round(3)
"""),
        md("## 6. Log to MLflow"),
        code("""
show(models.log_to_mlflow)
models.log_to_mlflow(results, lead)
"""),
    ],
)

# --------------------------------------------------------------------------- 07
write(
    "07_report.ipynb",
    [
        md("""
# 07 · Practice exams, cramming, and the report figures

Two last analyses (timed practice exams and timing of practice), then every figure in `outputs/figures/`,
the headline metrics, and the aggregate tables for Looker Studio.
"""),
        code(SETUP),
        code("""
from IPython.display import display
from so_ell import practice_exams, report
from so_ell.scores import build_windows
scores = pd.read_parquet(config.PROCESSED_DIR / "student_exam_scores.parquet")
windows = build_windows(pd.read_parquet(config.EXAM_ATTEMPTS_PARQUET))
"""),
        md("## 1. Timed practice exams\nStudents who finished a timed practice midterm vs those who started one or did none."),
        code("""
show(practice_exams.practice_midterm_analysis)
practice = practice_exams.practice_midterm_analysis(scores)
display(practice["group"].round(2))
practice["signals"].round(3)
"""),
        md("## 2. Cramming vs spacing\nShare of each student's last-28-day practice that fell in each window before the exam, by score group."),
        code("""
show(practice_exams.cramming_analysis)
cram = practice_exams.cramming_analysis(scores, windows)
cram["summary"].round(3)
"""),
        md("## 3. Build every figure and table\n`report.build_all` writes the practice-exam and cramming tables above, all figures, `outputs/headline_metrics.csv` and the Looker export."),
        code("""
show(report.build_all)
report.build_all()
pd.read_csv(config.OUTPUTS_DIR / "headline_metrics.csv").round(3)
"""),
        md("## 4. The figures"),
        code("""
from IPython.display import Markdown
# Linked rather than embedded, so the notebook stays small; GitHub renders the relative paths.
display(Markdown("\n\n".join(f"**{p.stem}**\n\n![{p.stem}](../outputs/figures/{p.name})" for p in sorted(config.FIGURES_DIR.glob("*.png")))))
"""),
    ],
)
