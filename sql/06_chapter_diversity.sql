-- How spread out was a student's practice across chapters?
-- 1 - Herfindahl, so 0 = all events in one chapter, closer to 1 = more even.

WITH chapter_counts AS (
    SELECT
        semester_raw,
        anon_student_id,
        midterm,
        level_chapter,
        COUNT(*) AS n_events
    FROM pre_exam_events
    WHERE level_chapter IS NOT NULL
      AND TRIM(level_chapter) NOT IN ('', 'py4e-int', 'ack', 'toctree', 'index')
    GROUP BY 1, 2, 3, 4
),
with_share AS (
    SELECT
        *,
        n_events / SUM(n_events) OVER (
            PARTITION BY semester_raw, anon_student_id, midterm
        ) AS chapter_share
    FROM chapter_counts
)
SELECT
    semester_raw,
    anon_student_id,
    midterm,
    1 - SUM(chapter_share * chapter_share) AS coverage__chapter_diversity
FROM with_share
GROUP BY semester_raw, anon_student_id, midterm;
