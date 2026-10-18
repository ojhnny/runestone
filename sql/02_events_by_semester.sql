-- Event volume by semester. Sanity check that every semester loaded (W20 and W21 are near-empty exports).

SELECT
    semester,
    COUNT(*) AS n_events,
    COUNT(DISTINCT anon_student_id) AS n_students,
    COUNT(DISTINCT DATE_TRUNC('day', time)) AS n_active_days
FROM raw_events
GROUP BY semester
ORDER BY semester;
