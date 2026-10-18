import pytest

from so_ell.exam_windows import is_real_exam_name, parse_midterm


@pytest.mark.parametrize(
    "name, midterm",
    [
        ("Midterm1-Fall21-Timed", "mid1"),
        ("Mid2-F25-Timed", "mid2"),
        ("Mid2-F25-Timed-Accom", "mid2"),
        ("m1 practice", "mid1"),
        ("Mid1-PI-Timed", "mid1"),
        ("middle-of-list", None),
        ("exam-review", None),
    ],
)
def test_parse_midterm(name, midterm):
    assert parse_midterm(name) == midterm


@pytest.mark.parametrize(
    "name, real",
    [
        ("Midterm1-Fall21-Timed", True),
        ("Mid2-F25-Timed-Accom", True),
        ("Mid2-F25-Practice-Timed", False),
        ("Mid1-PI-Timed", False),
        ("Mid2-Fall23-Practice-Timed-120-Min", False),
    ],
)
def test_real_exam_filter(name, real):
    assert is_real_exam_name(name) is real
