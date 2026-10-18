-- Family-level pre-exam volume. Complements the pandas quality features
-- (first/best score) which still need the action parsers.

SELECT
    semester_raw,
    anon_student_id,
    midterm,
    family,
    COUNT(*) AS total_events,
    COUNT(DISTINCT problem_name) AS unique_problems,
    COUNT(DISTINCT event_day) AS active_days,
    COUNT(DISTINCT session_id) AS unique_sessions
FROM pre_exam_events
GROUP BY 1, 2, 3, 4;
