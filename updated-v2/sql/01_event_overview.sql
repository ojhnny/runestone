-- How big is this log, and what's actually in it?
-- DuckDB reads the parquet in place, so this should be quick even at ~20M rows.

SELECT
    COUNT(*) AS n_events,
    COUNT(DISTINCT semester) AS n_semesters,
    COUNT(DISTINCT anon_student_id) AS n_students,
    COUNT(DISTINCT problem_name) AS n_problems,
    COUNT(DISTINCT session_id) AS n_sessions,
    MIN(time) AS first_event,
    MAX(time) AS last_event
FROM raw_events;
