-- Event volume by semester. Good sanity check that F21-F25 / W22-W24 all loaded.

SELECT
    semester,
    COUNT(*) AS n_events,
    COUNT(DISTINCT anon_student_id) AS n_students,
    COUNT(DISTINCT DATE_TRUNC('day', time)) AS n_active_days
FROM raw_events
GROUP BY semester
ORDER BY semester;
