"""Fill the ⟨placeholders⟩ in a markdown template from outputs/ tables.

    uv run python docs/fill_numbers.py docs/design_report.template.md docs/design_report.md

Keeps every number quoted in the report traceable to an output table.
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from so_ell import config

T = config.TABLES_DIR


def values() -> dict[str, str]:
    boot = pd.read_csv(T / "parsons_gap_bootstrap.csv").iloc[0]
    fe = pd.read_csv(T / "parsons_models.csv").set_index("spec")
    comp = pd.read_csv(T / "at_risk_model_comparison.csv").set_index("model")
    lead = pd.read_csv(T / "at_risk_lead_time.csv")
    corr = pd.read_csv(T / "feature_signals.csv").set_index("feature")
    sens = pd.read_csv(T / "key_threshold_sensitivity.csv")
    imp = pd.read_csv(T / "at_risk_importance_gbm.csv")
    pe = pd.read_csv(T / "practice_exam_signals.csv").set_index("metric")["value"]

    def auc_at(model: str, days: int) -> float:
        return float(lead.loc[lead["model"].eq(model) & lead["cutoff_days"].eq(days), "auc"].iloc[0])

    labels = {
        "cohort_fe": "Cohort fixed effects",
        "cohort_fe_plus_effort": "+ effort controls",
        "mid2_controls_for_mid1": "Midterm 2, controlling for midterm 1",
    }
    fe_rows = [
        f"| {labels[s]} | {int(r.n):,} | {10 * r.coef:.2f} | {10 * r.ci_low:.2f} to {10 * r.ci_high:.2f} |"
        for s, r in fe.iterrows()
    ]
    names = {
        "logistic": "Logistic, all exams",
        "gradient_boosting": "Gradient boosting, all exams",
        "mid2_mid1_only": "Midterm 2: midterm 1 score only",
        "mid2_practice_only": "Midterm 2: practice only",
        "mid2_mid1_plus_practice": "Midterm 2: midterm 1 + practice",
    }
    model_rows = [
        f"| {names[m]} | {r.auc:.3f} | {r.average_precision:.3f} | {r.brier:.3f} | {r.precision_top10pct:.2f} |"
        for m, r in comp.iterrows()
    ]
    p10 = comp.loc["logistic", "precision_top10pct"]
    return {
        "gap": f"{boot['estimate']:.1f}",
        "gap_r": f"{round(boot['estimate']):.0f}",
        "gap_lo": f"{boot['ci_low']:.1f}",
        "gap_hi": f"{boot['ci_high']:.1f}",
        "fe10": f"{10 * fe.loc['cohort_fe', 'coef']:.1f}",
        "fe10_effort": f"{10 * fe.loc['cohort_fe_plus_effort', 'coef']:.1f}",
        "fe10_mid1": f"{10 * fe.loc['mid2_controls_for_mid1', 'coef']:.1f}",
        "fe_table": "\n".join(fe_rows),
        "model_table": "\n".join(model_rows),
        "auc_log": f"{comp.loc['logistic', 'auc']:.2f}",
        "auc_gbm": f"{comp.loc['gradient_boosting', 'auc']:.2f}",
        "auc_m1": f"{comp.loc['mid2_mid1_only', 'auc']:.2f}",
        "auc_m1p": f"{comp.loc['mid2_mid1_plus_practice', 'auc']:.2f}",
        "auc14": f"{auc_at('logistic', 14):.2f}",
        "auc14_gbm": f"{auc_at('gradient_boosting', 14):.2f}",
        "p10": f"{p10:.0%}",
        "p10_pct": f"{p10:.0%}",
        "fp_share": f"{1 - p10:.0%}",
        "n_sig": f"{int((corr['q_value_bh'] < 0.05).sum())}",
        "n_feat": f"{len(corr)}",
        "r_parsons": f"{corr.loc['parsons__mean_first_score', 'within_group_rank_corr']:.2f}",
        "r_events": f"{corr.loc['all_activity__total_events', 'within_group_rank_corr']:.2f}",
        "r_days": f"{corr.loc['all_activity__active_days', 'within_group_rank_corr']:.2f}",
        "sens_lo": f"{sens['parsons_quartile_gap'].min():.1f}",
        "sens_hi": f"{sens['parsons_quartile_gap'].max():.1f}",
        "imp_parsons": f"{imp.iloc[0]['auc_drop_mean']:.3f}",
        "imp_next": f"{imp.iloc[1]['auc_drop_mean']:.3f}",
        "pe_gap": f"{pe['finished_vs_none_gap']:.1f}",
        "pe_corr": f"{pe['practice_score_rank_corr']:.2f}",
    }


def main(src: str, dst: str) -> None:
    text = open(src, encoding="utf-8").read()
    vals = values()
    missing = sorted(set(re.findall(r"⟨(\w+)⟩", text)) - set(vals))
    if missing:
        raise SystemExit(f"no value for: {missing}")
    text = re.sub(r"⟨(\w+)⟩", lambda m: vals[m.group(1)], text)
    open(dst, "w", encoding="utf-8").write(text)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
