"""Run every data stage on a tiny synthetic semester and check the parts that
went wrong in the original analysis:

- two section exports that both number students from 1 (ID collision)
- midterm 1 answers leaking into midterm 2 "practice" features
- scores reconstructed from in-exam answers
"""

from __future__ import annotations

import gzip
from datetime import datetime, timedelta

import pandas as pd
import pytest

from so_ell import config, ingest, pipeline

HEADER = [
    "Time", "Anon Student Id", "Action", "Problem Name", "Class", "Level (Chapter)", "Level (SubChapter)",
    "Session Id", "Problem View", "Selection", "Input", "Feedback Text", "Feedback Classification",
    "CF (Code)", "CF (Week No)", "CF (Institution)",
]
MID1 = datetime(2025, 10, 1, 18, 0)
MID2 = datetime(2025, 10, 29, 18, 0)


def _row(t, student, action, problem, session, selection, chapter="lists", inp=""):
    return [t.strftime("%Y-%m-%d %H:%M:%S.%f"), str(student), action, problem, "1", chapter, "sub",
            str(session), "1", selection, inp, "", "", "1", "5", "university"]


def _section_rows(strong_students: set[int]) -> list[list[str]]:
    rows = []
    for s in range(1, 5):
        good = s in strong_students
        # Practice before midterm 1: two Parsons problems, first try right only for strong students.
        for k, day in enumerate([10, 5]):
            t = MID1 - timedelta(days=day)
            rows.append(_row(t, s, "correct:1" if good else "incorrect:1", f"practice_parsons_{k}", 100 + s, "parsons"))
            rows.append(_row(t + timedelta(minutes=1), s, "correct:2", f"practice_parsons_{k}", 100 + s, "parsons"))
        # Midterm 1 (timed). Strong students get both items right.
        sess = 200 + s
        rows.append(_row(MID1, s, "start", "Mid1-F25-Timed", sess, "timedExam"))
        rows.append(_row(MID1 + timedelta(minutes=5), s, "answer:1:correct" if good else "answer:2:no", "exam1_mc", sess, "mChoice"))
        rows.append(_row(MID1 + timedelta(minutes=9), s, "correct:1" if good else "incorrect:1", "exam1_parsons", sess, "parsons"))
        rows.append(_row(MID1 + timedelta(minutes=60), s, "finish", "Mid1-F25-Timed", sess, "timedExam"))
        # A practice exam that must NOT be treated as the real midterm.
        rows.append(_row(MID2 - timedelta(days=3), s, "start", "Mid2-F25-Practice-Timed", 300 + s, "timedExam"))
        rows.append(_row(MID2 - timedelta(days=3, minutes=-30), s, "finish", "Mid2-F25-Practice-Timed", 300 + s, "timedExam"))
        # Midterm 2.
        sess = 400 + s
        rows.append(_row(MID2, s, "start", "Mid2-F25-Timed", sess, "timedExam"))
        rows.append(_row(MID2 + timedelta(minutes=5), s, "answer:1:correct" if good else "answer:2:no", "exam2_mc", sess, "mChoice"))
        rows.append(_row(MID2 + timedelta(minutes=60), s, "finish", "Mid2-F25-Timed", sess, "timedExam"))
    return rows


def _write_export(path, rows):
    text = "\t".join(HEADER) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
    with gzip.open(path, "wt") as handle:
        handle.write(text)


@pytest.fixture(scope="module")
def built():
    raw = config.RAW_DIR / "F25"
    raw.mkdir(parents=True, exist_ok=True)
    # Same local ids 1-4 in both sections; MW's strong students are 1, 2 and TTh's are 3, 4.
    _write_export(raw / "F25-SI201-MW_datashop.tab.gz", _section_rows({1, 2}))
    _write_export(raw / "F25-SI201-TTh_datashop.tab.gz", _section_rows({3, 4}))
    ingest.build_manifest()
    config.ensure_dirs()
    pipeline.stage_ingest()
    pipeline.stage_exam_windows()
    pipeline.stage_scores()
    pipeline.stage_features()
    return {
        "events": pd.read_parquet(config.EVENTS_PARQUET),
        "scores": pd.read_parquet(pipeline.SCORES),
        "features": pd.read_parquet(pipeline.FEATURES),
    }


def test_section_ids_are_namespaced(built):
    ids = built["events"]["anon_student_id"].unique()
    assert len(ids) == 8, "4 students x 2 sections must stay 8 distinct people"
    assert "F25-MW-0001" in ids and "F25-TTh-0001" in ids


def test_practice_exam_is_not_the_real_midterm(built):
    attempts = pd.read_parquet(config.EXAM_ATTEMPTS_PARQUET)
    assert set(attempts["anchor_problem_name"]) == {"Mid1-F25-Timed", "Mid2-F25-Timed"}
    assert len(attempts) == 16


def test_scores_follow_in_exam_answers(built):
    s = built["scores"].set_index(["anon_student_id", "midterm"])["score_pct"]
    assert s[("F25-MW-0001", "mid1")] == 100
    assert s[("F25-MW-0003", "mid1")] == 0
    assert s[("F25-TTh-0003", "mid1")] == 100  # would be averaged with MW-3 if ids collided
    assert s[("F25-TTh-0001", "mid2")] == 0


def test_midterm1_answers_do_not_leak_into_midterm2_features(built):
    f = built["features"].set_index(["anon_student_id", "midterm"])
    # Practice Parsons problems are practice_parsons_0/1. exam1_parsons happened
    # inside the mid1 window and must not count, in pandas or in SQL features.
    assert f.loc[("F25-MW-0001", "mid2"), "parsons__unique_problems"] == 2
    assert f.loc[("F25-MW-0001", "mid2"), "coverage__unique_problems"] == 2


def test_first_try_feature_separates_strong_students(built):
    f = built["features"].set_index(["anon_student_id", "midterm"])
    assert f.loc[("F25-MW-0001", "mid1"), "parsons__mean_first_score"] == 1.0
    assert f.loc[("F25-MW-0004", "mid1"), "parsons__mean_first_score"] == 0.0
