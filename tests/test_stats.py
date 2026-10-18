import numpy as np
import pandas as pd

from so_ell.stats import (
    add_cohort,
    benjamini_hochberg,
    bootstrap_quartile_gap,
    compute_corr_table,
    within_group_quartile_gap,
)


def _two_cohorts() -> pd.DataFrame:
    # Two cohorts 40 points apart. Inside each, the feature lines up with the
    # score, so the within-cohort gap is the same (30 points) in both.
    feat = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    high = [50, 51, 52, 60, 61, 62, 70, 71, 72, 80, 81, 82]
    low = [s - 40 for s in high]
    return add_cohort(
        pd.DataFrame(
            {
                "semester_raw": ["F21"] * 12 + ["F22"] * 12,
                "anon_student_id": [f"s{i}" for i in range(24)],
                "midterm": ["mid1"] * 24,
                "feature": feat * 2,
                "score_pct": high + low,
            }
        )
    )


def test_quartile_gap_ignores_cohort_shifts():
    gap = within_group_quartile_gap(_two_cohorts(), "cohort", "feature", "score_pct")
    assert 29 < gap < 31


def test_corr_table_sorts_strongest_first_and_adds_q_values():
    df = pd.DataFrame(
        {
            "cohort": ["A"] * 12,
            "good": list(range(12)),
            "noise": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "score_pct": list(range(12)),
        }
    )
    out = compute_corr_table(df, "cohort", "score_pct", ["good", "noise"])
    assert out.iloc[0]["feature"] == "good"
    assert {"p_value", "q_value_bh"} <= set(out.columns)


def test_benjamini_hochberg_matches_hand_calculation():
    p = pd.Series([0.01, 0.04, 0.03, 0.20])
    q = benjamini_hochberg(p)
    # sorted p: .01 .03 .04 .20 -> p*m/rank: .04 .06 .0533 .20 -> monotone: .04 .0533 .0533 .20
    assert np.allclose(q.to_numpy(), [0.04, 0.05333, 0.05333, 0.20], atol=1e-4)


def test_bootstrap_interval_contains_estimate():
    df = _two_cohorts().rename(columns={"feature": "parsons__mean_first_score"})
    df = pd.concat([df] * 3, ignore_index=True)
    df["anon_student_id"] = [f"s{i}" for i in range(len(df))]
    out = bootstrap_quartile_gap(df, n_boot=200)
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]
