from so_ell.scoring import (
    activity_family,
    parse_hparsons_score,
    parse_mchoice_score,
    parse_parsons_score,
    parse_unittest_pct,
)


def test_parsons_correct_and_incorrect():
    assert parse_parsons_score("correct:1:ok") == 1.0
    assert parse_parsons_score("incorrect:3:moved") == 0.0
    assert pd_isna(parse_parsons_score("move:block-3"))


def test_mchoice_flags():
    assert parse_mchoice_score("answer:2:correct") == 1.0
    assert parse_mchoice_score("answer:1:no") == 0.0


def test_unittest_percent():
    assert parse_unittest_pct("percent:80.0:passed") == 0.8
    assert pd_isna(parse_unittest_pct("ran without tests"))


def test_hparsons_json():
    assert parse_hparsons_score('{"correct": "true", "percent": 1}') == 1.0
    assert parse_hparsons_score('{"percent": 0.5}') == 0.5


def test_activity_family_lumps_the_parsons_variants():
    assert activity_family("parsonsMove") == "parsons"
    assert activity_family("hparsonsAnswer") == "parsons"
    assert activity_family("unittest") == "activecode"
    assert activity_family("selectquestion") == "conceptcheck"
    assert activity_family("page") == "other"


def pd_isna(value) -> bool:
    import math

    return value != value or (isinstance(value, float) and math.isnan(value))
