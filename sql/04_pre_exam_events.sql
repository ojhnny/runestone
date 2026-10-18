-- Pre-exam practice events joined onto each student's exam window.
--
-- Leakage guards:
--   1. Only events strictly before this exam's start.
--   2. Events inside ANY of the student's timed exam windows are dropped
--      (the practice_events view). Without this, a student's midterm 1 answers
--      would count as "practice" for midterm 2. The first version of this
--      file only had guard 1.
-- days_before is 1 on the day before the exam, 7 a week before, etc.
-- Optional: SET VARIABLE cutoff_days = 14; keeps only practice at least
-- 14 days before the exam (used by the early-warning analysis).

CREATE OR REPLACE VIEW exam_windows AS
SELECT
    semester_raw,
    anon_student_id,
    midterm,
    ANY_VALUE(semester) AS semester,
    MIN(attempt_start) AS exam_start,
    MAX(attempt_end) AS exam_end
FROM exam_attempts
GROUP BY semester_raw, anon_student_id, midterm;

CREATE OR REPLACE VIEW practice_events AS
SELECT e.*
FROM raw_events e
WHERE e.time IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM exam_windows x
      WHERE x.semester_raw = e.semester
        AND x.anon_student_id = e.anon_student_id
        AND e.time BETWEEN x.exam_start AND x.exam_end
  );

CREATE OR REPLACE VIEW pre_exam_events AS
SELECT
    e.semester AS semester_raw,
    e.anon_student_id,
    w.midterm,
    e.time,
    CAST(e.time AS DATE) AS event_day,
    e.session_id,
    e.problem_name,
    e.level_chapter,
    e.level_subchapter,
    e.selection,
    e.cf_week_no,
    DATE_DIFF('day', CAST(e.time AS DATE), CAST(w.exam_start AS DATE)) AS days_before,
    CASE e.selection
        WHEN 'mChoice' THEN 'mchoice'
        WHEN 'parsonsMove' THEN 'parsons'
        WHEN 'parsons' THEN 'parsons'
        WHEN 'hparsons' THEN 'parsons'
        WHEN 'hparsonsAnswer' THEN 'parsons'
        WHEN 'activecode' THEN 'activecode'
        WHEN 'unittest' THEN 'activecode'
        WHEN 'ac_error' THEN 'activecode'
        WHEN 'fillb' THEN 'fillb'
        WHEN 'clickableArea' THEN 'clickable'
        WHEN 'dragNdrop' THEN 'dragndrop'
        WHEN 'dragNdrop-drop' THEN 'dragndrop'
        WHEN 'shortanswer' THEN 'shortanswer'
        WHEN 'selectquestion' THEN 'conceptcheck'
        ELSE 'other'
    END AS family
FROM practice_events e
INNER JOIN exam_windows w
    ON e.semester = w.semester_raw
   AND e.anon_student_id = w.anon_student_id
WHERE e.time < w.exam_start - to_days(CAST(COALESCE(getvariable('cutoff_days'), 0) AS INTEGER))
  AND e.selection IN (
      'mChoice', 'fillb', 'parsonsMove', 'parsons', 'activecode',
      'unittest', 'ac_error', 'hparsons', 'hparsonsAnswer',
      'clickableArea', 'dragNdrop', 'dragNdrop-drop',
      'shortanswer', 'selectquestion'
  );

SELECT COUNT(*) AS n_pre_exam_events
FROM pre_exam_events;
