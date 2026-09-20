-- Timed exam windows are already in exam_windows/. We just collapse
-- extra sessions into a single window per student / semester / midterm.

SELECT
    semester_raw,
    anon_student_id,
    midterm,
    ANY_VALUE(semester) AS semester,
    ANY_VALUE(normalized_midterm_name) AS normalized_midterm_name,
    MIN(attempt_start) AS exam_start,
    MAX(attempt_end) AS exam_end,
    COUNT(DISTINCT session_id) AS n_exam_sessions,
    SUM(attempt_minutes) AS total_exam_minutes
FROM exam_attempts
GROUP BY semester_raw, anon_student_id, midterm;
