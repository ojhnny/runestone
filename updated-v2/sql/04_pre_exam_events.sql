-- Pre-exam practice events joined onto each student's exam window.
--
-- Important: we drop events that fall inside the exam itself, otherwise
-- "practice quality" would leak the test. days_before is 1 on the day
-- before the exam, 7 a week before, etc.

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
FROM raw_events e
INNER JOIN exam_windows w
    ON e.semester = w.semester_raw
   AND e.anon_student_id = w.anon_student_id
WHERE e.time < w.exam_start
  AND e.selection IN (
      'mChoice', 'fillb', 'parsonsMove', 'parsons', 'activecode',
      'unittest', 'ac_error', 'hparsons', 'hparsonsAnswer',
      'clickableArea', 'dragNdrop', 'dragNdrop-drop',
      'shortanswer', 'selectquestion'
  );

SELECT COUNT(*) AS n_pre_exam_events
FROM pre_exam_events;
