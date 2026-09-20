"""Parse Runestone event logs into scores.

The ebook dumps a lot of different activity types into one `action` column,
so each family needs its own little parser. I kept these picky on purpose:
if we can't tell whether an event was correct, leave it as missing instead
of guessing.
"""

from __future__ import annotations

import ast
import json
import re

import numpy as np
import pandas as pd

WINDOW_KEYS = ["semester_raw", "anon_student_id", "midterm"]

# These don't come with a true/false flag. We only score them if enough
# students land on the same final answer that it looks like a key.
KEYED_FAMILIES = {"fillb", "clickable", "dragndrop"}
EXPLICIT_SCORE_FAMILIES = {"mchoice", "parsons", "activecode"}
SCOREABLE_FAMILIES = KEYED_FAMILIES | EXPLICIT_SCORE_FAMILIES

# Nav / boilerplate chapters that aren't really course topics.
SKIP_CHAPTERS = {"py4e-int", "ack", "toctree", "index", ""}

EXAM_SELECTIONS = [
    "mChoice",
    "fillb",
    "parsonsMove",
    "parsons",
    "activecode",
    "unittest",
    "ac_error",
    "hparsons",
    "hparsonsAnswer",
    "clickableArea",
    "dragNdrop",
    "dragNdrop-drop",
    "shortanswer",
    "selectquestion",
]


def parse_unittest_pct(action: object) -> float:
    # ActiveCode stores something like "percent:80.0" when unit tests run.
    match = re.search(r"percent:([0-9.]+)", str(action))
    return float(match.group(1)) / 100 if match else np.nan


def parse_mchoice_score(action: object) -> float:
    text = str(action)
    if ":correct" in text:
        return 1.0
    if ":no" in text:
        return 0.0
    return np.nan


def parse_parsons_score(action: object) -> float:
    # Regular Parsons checks look like "correct:..." / "incorrect:...".
    text = str(action)
    if text.startswith("correct"):
        return 1.0
    if text.startswith("incorrect"):
        return 0.0
    return np.nan


def parse_hparsons_score(action: object) -> float:
    # Horizontal Parsons usually logs a JSON blob. Sometimes it's just text.
    try:
        obj = json.loads(str(action))
    except Exception:
        obj = None
    if isinstance(obj, dict):
        if "percent" in obj:
            try:
                return float(obj["percent"])
            except Exception:
                return np.nan
        if "correct" in obj:
            return 1.0 if str(obj["correct"]).upper().startswith("T") else 0.0
    match = re.search(r'"percent"\s*:\s*([0-9.]+)', str(action))
    return float(match.group(1)) if match else np.nan


def normalize_response(value: object) -> str | None:
    """Smash fill-in / click / drag answers into a comparable string."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    value_to_use = parsed if parsed is not None else text
    if isinstance(value_to_use, dict):
        normalized = {}
        for key, val in sorted(value_to_use.items()):
            if isinstance(val, list):
                normalized[str(key).strip().lower()] = sorted(str(item).strip().lower() for item in val)
            else:
                normalized[str(key).strip().lower()] = str(val).strip().lower()
        return json.dumps(normalized, sort_keys=True)
    if isinstance(value_to_use, (list, tuple, set)):
        return json.dumps([str(item).strip().lower() for item in value_to_use])
    compact = re.sub(r"\s+", " ", str(value_to_use).strip().strip("\"'").lower())
    return compact or None


def activity_family(selection: object) -> str:
    mapping = {
        "mChoice": "mchoice",
        "parsonsMove": "parsons",
        "parsons": "parsons",
        "hparsons": "parsons",
        "hparsonsAnswer": "parsons",
        "activecode": "activecode",
        "unittest": "activecode",
        "ac_error": "activecode",
        "fillb": "fillb",
        "clickableArea": "clickable",
        "dragNdrop": "dragndrop",
        "dragNdrop-drop": "dragndrop",
        "shortanswer": "shortanswer",
        "selectquestion": "conceptcheck",
    }
    return mapping.get(str(selection), "other")


def first_non_null(series: pd.Series) -> float:
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def last_non_null(series: pd.Series) -> float:
    non_null = series.dropna()
    return non_null.iloc[-1] if not non_null.empty else np.nan
