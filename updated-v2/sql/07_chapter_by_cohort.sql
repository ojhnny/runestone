-- Aggregate chapter mix by cohort. This one is Looker-friendly (no student ids).

SELECT
    semester_raw,
    midterm,
    level_chapter AS chapter,
    COUNT(*) AS n_events,
    COUNT(DISTINCT anon_student_id) AS n_students,
    COUNT(DISTINCT problem_name) AS n_problems
FROM pre_exam_events
WHERE level_chapter IS NOT NULL
  AND TRIM(level_chapter) NOT IN ('', 'py4e-int', 'ack', 'toctree', 'index')
GROUP BY semester_raw, midterm, level_chapter
ORDER BY semester_raw, midterm, n_events DESC;
