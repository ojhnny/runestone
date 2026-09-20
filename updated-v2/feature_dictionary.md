# Pre-exam features

Names use `family__metric` so they plug into the within-cohort correlation helper.

## From the pandas scoring pass (`rebuild_midterm_scores.py`)

Quality: `mean_first_score`, `mean_last_score`, `mean_best_score`, `mean_score_gain`, `avg_event_score`

Volume: `total_events`, `unique_problems`, `problems_seen`, `scored_events`, `scored_problems`

Consistency: `active_days`, `mean_events_per_problem`

Families: `parsons`, `activecode`, `mchoice`, `fillb`, `clickable`, `dragndrop`, `shortanswer`, `conceptcheck`, plus `all_activity` and `learning`.

## Added in DuckDB (`sql/05_student_exam_features.sql` and `06_chapter_diversity.sql`)

Coverage

- `coverage__unique_chapters`
- `coverage__unique_subchapters`
- `coverage__chapter_diversity` (1 - Herfindahl; higher = more even mix of chapters)
- `coverage__family_count`
- `coverage__unique_problems`
- `coverage__total_events`

Consistency

- `consistency__active_days`
- `consistency__span_days`
- `consistency__unique_sessions`
- `consistency__unique_weeks`
- `consistency__problems_per_active_day`
- `consistency__events_per_active_day`

Timing

- `timing__first_days_before`
- `timing__last_days_before`
- `timing__events_last_3d` / `_7d` / `_14d` / `_28d`
- `timing__share_last_3d` / `_7d` / `_14d`

Mix

- `mix__parsons_share`
- `mix__activecode_share`
- `mix__coding_share`
- `mix__mchoice_share`
- `mix__conceptcheck_share`
