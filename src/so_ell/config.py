"""Paths and analysis constants.

Every path the pipeline reads or writes is defined here, so nothing else in
the package hard-codes a location. Set SO_ELL_ROOT to run against a copy of
the data somewhere else (the tests do this with a tiny synthetic dataset).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SO_ELL_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_MANIFEST = RAW_DIR / "MANIFEST.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

EVENTS_PARQUET = INTERIM_DIR / "events.parquet"
EXAM_ATTEMPTS_PARQUET = INTERIM_DIR / "exam_attempts.parquet"

SQL_DIR = ROOT / "sql"

OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOOKER_DIR = OUTPUTS_DIR / "looker_studio"
QUALITY_DIR = OUTPUTS_DIR / "data_quality"

MLRUNS_DIR = ROOT / "mlruns"

# Semesters whose Datashop export actually contains a course. W20 (404 rows)
# and W21 (5 rows) are near-empty exports with no timed midterms; they are
# ingested so the row counts reconcile, then dropped from the analysis.
ANALYSIS_SEMESTERS = ["F21", "W22", "F22", "W23", "F23", "W24", "F24", "F25"]
EXCLUDED_SEMESTERS = {"W20": "export has 404 rows and no timed exams", "W21": "export has 5 rows"}

# Student-exam grain used everywhere downstream.
WINDOW_KEYS = ["semester_raw", "anon_student_id", "midterm"]

# Inferred answer keys (fill-in, clickable, drag-and-drop) are only trusted when
# at least this share of students ended on the same final answer.
KEY_CONFIDENCE_THRESHOLD = 0.65

RANDOM_SEED = 206


def ensure_dirs() -> None:
    for path in [INTERIM_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, LOOKER_DIR, QUALITY_DIR]:
        path.mkdir(parents=True, exist_ok=True)
