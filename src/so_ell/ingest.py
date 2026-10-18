"""Raw Datashop exports -> one clean event log.

Runestone exports each course section as a tab-separated "Datashop" file.
This step reads every file listed in data/raw/MANIFEST.csv, checks its
checksum, standardizes the column names, and writes data/interim/events.parquet.

The one non-obvious transformation is the student key. Each section export
numbers its students from 1, so in semesters with two sections (F23, F24, F25)
"student 5" in the Monday/Wednesday file and "student 5" in the Tuesday/Thursday
file are different people. The original analysis keyed on (semester, id) and
silently merged them. Here every id is namespaced by semester and section:
"F25-MW-0005". Session ids get the same treatment.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import duckdb
import pandas as pd

from so_ell import config

CANONICAL_COLUMNS = [
    "time",
    "anon_student_id",
    "action",
    "problem_name",
    "class",
    "level_chapter",
    "level_subchapter",
    "session_id",
    "problem_view",
    "selection",
    "input",
    "feedback_text",
    "feedback_classification",
    "cf_code",
    "cf_week_no",
    "cf_institution",
]
INT_COLUMNS = ["class", "problem_view", "cf_week_no"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def infer_section(file_name: str) -> str:
    name = file_name.lower()
    if re.search(r"(^|[-_])mw([-_]|$)", name):
        return "MW"
    if "tth" in name or "t-thurs" in name:
        return "TTh"
    return "ALL"


def build_manifest(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Scan data/raw/<semester>/ and describe every compressed export.

    Run once (``so-ell manifest``) when raw files change, then commit the CSV.
    """
    rows = []
    for path in sorted(raw_dir.glob("*/*")):
        if path.suffix not in {".gz", ".zip", ".bz2"}:
            continue
        semester = path.parent.name
        rows.append(
            {
                "semester": semester,
                "section": infer_section(path.name),
                "path": path.relative_to(raw_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(raw_dir / "MANIFEST.csv", index=False)
    return manifest


def standardize_columns(columns: pd.Index) -> list[str]:
    return (
        columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .tolist()
    )


def read_export(path: Path, semester: str, section: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True, compression="infer")
    df.columns = standardize_columns(df.columns)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    for col in missing:
        df[col] = pd.NA
    df = df[CANONICAL_COLUMNS].copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    local_student = pd.to_numeric(df["anon_student_id"], errors="coerce").astype("Int64")
    local_session = pd.to_numeric(df["session_id"], errors="coerce").astype("Int64")
    prefix = f"{semester}-{section}-"
    df["local_student_id"] = local_student
    df["anon_student_id"] = (prefix + local_student.astype("string").str.zfill(4)).astype("string")
    df["session_id"] = (prefix + local_session.astype("string")).astype("string")
    df["semester"] = semester
    df["section"] = section
    df["source_file"] = path.name
    return df


def run_ingest(
    raw_dir: Path = config.RAW_DIR,
    out_path: Path = config.EVENTS_PARQUET,
    verify_checksums: bool = True,
) -> pd.DataFrame:
    """Read every manifest entry and write the combined event log.

    Returns a per-file row-count table (also written to outputs/data_quality).
    """
    manifest = pd.read_csv(raw_dir / "MANIFEST.csv")
    parts_dir = out_path.parent / "_event_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for row in manifest.itertuples(index=False):
        path = raw_dir / row.path
        if not path.exists():
            raise FileNotFoundError(f"{path} is listed in MANIFEST.csv but missing. See data/README.md.")
        if verify_checksums and sha256(path) != row.sha256:
            raise ValueError(f"Checksum mismatch for {path}; the raw export changed.")
        print(f"  reading {row.path}", flush=True)
        df = read_export(path, row.semester, row.section)
        part = parts_dir / f"{row.semester}_{row.section}.parquet"
        df.to_parquet(part, index=False)
        counts.append(
            {
                "semester": row.semester,
                "section": row.section,
                "source_file": path.name,
                "rows": len(df),
                "students": df["anon_student_id"].nunique(),
                "null_time_rows": int(df["time"].isna().sum()),
            }
        )
        del df

    # DuckDB does the concat + sort out of core, so this works on a laptop.
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    parts_glob = (parts_dir / "*.parquet").as_posix()
    target = out_path.as_posix()
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet('{parts_glob}', union_by_name = true)
            ORDER BY semester, anon_student_id, time
        ) TO '{target}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)
        """
    )
    con.close()
    for part in parts_dir.glob("*.parquet"):
        part.unlink()
    parts_dir.rmdir()
    return pd.DataFrame(counts)
