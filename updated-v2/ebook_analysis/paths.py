from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "analysis_outputs"
SQL_DIR = BASE_DIR / "sql"
LOOKER_DIR = BASE_DIR / "looker_studio"
PRIVATE_LOOKER_DIR = LOOKER_DIR / "private"
EVENTS_PARQUET = BASE_DIR / "runestone_event_log.parquet"
ATTEMPTS_PARQUET = BASE_DIR / "exam_windows" / "timed_exam_attempts.parquet"


def ensure_output_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    LOOKER_DIR.mkdir(exist_ok=True)
    PRIVATE_LOOKER_DIR.mkdir(parents=True, exist_ok=True)
