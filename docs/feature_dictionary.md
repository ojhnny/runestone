# Feature dictionary

One row of `data/processed/student_exam_features.parquet` = one student × one midterm. Every feature uses only practice **before that exam started**, with events inside any timed exam window removed. Names are `family__metric`.

## Identifiers and target

| Column | Meaning |
|---|---|
| `semester_raw` | F21, W22, … F25 |
| `anon_student_id` | `<semester>-<section>-<local id>`, e.g. `F25-MW-0005` (section-namespaced, see design report §4) |
| `midterm` | `mid1` or `mid2` |
| `cohort` | `<semester>_<midterm>`; all comparisons are within a cohort |
| `score_pct` | reconstructed midterm score, 0–100 (mean over scorable items) |
| `coverage_pct` | % of the items the student touched during the exam that could be scored |

## Families (pandas pass, `src/so_ell/features.py`)

`parsons`, `activecode`, `mchoice`, `fillb`, `clickable`, `dragndrop`, `shortanswer`, `conceptcheck`, plus roll-ups `all_activity` (every practice event) and `learning` (every scorable problem).

| Metric | Meaning |
|---|---|
| `total_events` | practice events of this family |
| `active_days` | distinct calendar days with this family |
| `unique_problems` | distinct problems touched |
| `scored_events` | events that carry a correctness signal |
| `avg_event_score` | mean correctness over every scored event (rewards getting it right, penalizes repeated wrong checks) |
| `problems_seen`, `scored_problems` | problems opened / problems with a score |
| `mean_first_score` | mean over problems of the **first scored attempt** (the headline feature for Parsons) |
| `mean_last_score`, `mean_best_score` | mean over problems of last / best attempt |
| `mean_score_gain` | mean of last − first; negative association with exam score is a baseline effect (design report §8.4) |
| `mean_events_per_problem` | persistence / struggle per problem |

Counts are 0, not missing, when a student never used a family. Quality metrics stay missing when there is nothing to average.

## DuckDB features (`sql/05_student_exam_features.sql`, `sql/06_chapter_diversity.sql`)

| Column | Meaning |
|---|---|
| `coverage__total_events`, `coverage__unique_problems` | across all answer-type activity |
| `coverage__unique_chapters`, `coverage__unique_subchapters` | topic breadth (navigation chapters excluded) |
| `coverage__family_count` | how many activity types used |
| `coverage__chapter_diversity` | 1 − Herfindahl index over chapters; 0 = all in one chapter |
| `consistency__active_days`, `__unique_sessions`, `__unique_weeks` | spread of practice |
| `consistency__span_days` | first to last practice day |
| `consistency__problems_per_active_day`, `__events_per_active_day` | intensity per active day |
| `timing__first_days_before`, `__last_days_before` | how early practice started / how close to the exam it ended |
| `timing__events_last_{3,7,14,28}d`, `timing__share_last_{3,7,14}d` | cramming measures |
| `mix__{parsons,activecode,coding,mchoice,conceptcheck}_share` | composition of practice |

## Early-warning variants

`data/processed/student_exam_features_cutoff_{N}d.parquet` recompute every feature using only practice at least N days before the exam (N = 3, 7, 14, 21, 28). Used by `outputs/tables/at_risk_lead_time.csv`.
