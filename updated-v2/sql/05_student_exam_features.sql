-- Student-exam features that are easier in SQL than in pandas:
-- topic coverage, spacing / cramming, session mix, activity mix.
--
-- Column names use family__metric so they drop into the existing
-- within-cohort correlation code.

SELECT
    semester_raw,
    anon_student_id,
    midterm,

    COUNT(*) AS coverage__total_events,
    COUNT(DISTINCT problem_name) AS coverage__unique_problems,
    COUNT(DISTINCT CASE
        WHEN level_chapter IS NOT NULL
         AND TRIM(level_chapter) NOT IN ('', 'py4e-int', 'ack', 'toctree', 'index')
        THEN level_chapter
    END) AS coverage__unique_chapters,
    COUNT(DISTINCT CASE
        WHEN level_subchapter IS NOT NULL
         AND TRIM(level_subchapter) NOT IN ('', 'index', 'toctree')
        THEN level_subchapter
    END) AS coverage__unique_subchapters,
    COUNT(DISTINCT family) AS coverage__family_count,

    COUNT(DISTINCT event_day) AS consistency__active_days,
    COUNT(DISTINCT session_id) AS consistency__unique_sessions,
    COUNT(DISTINCT cf_week_no) AS consistency__unique_weeks,
    DATE_DIFF('day', MIN(event_day), MAX(event_day)) AS consistency__span_days,
    COUNT(DISTINCT problem_name)::DOUBLE
        / NULLIF(COUNT(DISTINCT event_day), 0) AS consistency__problems_per_active_day,
    COUNT(*)::DOUBLE
        / NULLIF(COUNT(DISTINCT event_day), 0) AS consistency__events_per_active_day,

    MAX(days_before) AS timing__first_days_before,
    MIN(days_before) AS timing__last_days_before,
    SUM(CASE WHEN days_before BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS timing__events_last_3d,
    SUM(CASE WHEN days_before BETWEEN 1 AND 7 THEN 1 ELSE 0 END) AS timing__events_last_7d,
    SUM(CASE WHEN days_before BETWEEN 1 AND 14 THEN 1 ELSE 0 END) AS timing__events_last_14d,
    SUM(CASE WHEN days_before BETWEEN 1 AND 28 THEN 1 ELSE 0 END) AS timing__events_last_28d,
    SUM(CASE WHEN days_before BETWEEN 1 AND 3 THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(*), 0) AS timing__share_last_3d,
    SUM(CASE WHEN days_before BETWEEN 1 AND 7 THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(*), 0) AS timing__share_last_7d,
    SUM(CASE WHEN days_before BETWEEN 1 AND 14 THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(*), 0) AS timing__share_last_14d,

    SUM(CASE WHEN family = 'parsons' THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(*), 0) AS mix__parsons_share,
    SUM(CASE WHEN family = 'activecode' THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(*), 0) AS mix__activecode_share,
    SUM(CASE WHEN family IN ('parsons', 'activecode') THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(*), 0) AS mix__coding_share,
    SUM(CASE WHEN family = 'mchoice' THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(*), 0) AS mix__mchoice_share,
    SUM(CASE WHEN family = 'conceptcheck' THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(*), 0) AS mix__conceptcheck_share
FROM pre_exam_events
GROUP BY semester_raw, anon_student_id, midterm;
