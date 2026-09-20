-- Last-28-day timing bins, still at the student-exam grain.
-- We collapse this to score bands later so Looker doesn't need student ids.

SELECT
    semester_raw,
    anon_student_id,
    midterm,
    CASE
        WHEN days_before BETWEEN 1 AND 3 THEN '1-3 days before'
        WHEN days_before BETWEEN 4 AND 7 THEN '4-7 days before'
        WHEN days_before BETWEEN 8 AND 14 THEN '8-14 days before'
        WHEN days_before BETWEEN 15 AND 28 THEN '15-28 days before'
    END AS timing_bin,
    COUNT(*) AS n_events
FROM pre_exam_events
WHERE days_before BETWEEN 1 AND 28
GROUP BY 1, 2, 3, 4;
