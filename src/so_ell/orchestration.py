"""Dagster asset graph for the pipeline.

Each asset wraps one pipeline stage and declares which stage it depends on,
so Dagster can draw the lineage graph, show when each table was last built,
and re-materialize only what is downstream of a change.

    uv sync --extra orchestration
    uv run dagster dev        # then open http://localhost:3000

The CLI (``so-ell all``) runs the same functions without Dagster.
"""

from __future__ import annotations

import pandas as pd
from dagster import AssetExecutionContext, Definitions, MaterializeResult, MetadataValue, asset

from so_ell import config, pipeline


def _rows(path) -> int:
    return len(pd.read_parquet(path, columns=None)) if path.exists() else 0


@asset(group_name="ingest", description="All raw Datashop exports combined into one event log.")
def event_log(context: AssetExecutionContext) -> MaterializeResult:
    pipeline.stage_ingest()
    counts = pd.read_csv(config.QUALITY_DIR / "ingest_row_counts.csv")
    return MaterializeResult(
        metadata={
            "rows": int(counts["rows"].sum()),
            "files": len(counts),
            "row_counts": MetadataValue.md(counts.to_markdown(index=False)),
        }
    )


@asset(deps=[event_log], group_name="ingest", description="Real timed midterm attempts per student.")
def exam_attempts() -> MaterializeResult:
    pipeline.stage_exam_windows()
    return MaterializeResult(metadata={"attempts": _rows(config.EXAM_ATTEMPTS_PARQUET)})


@asset(deps=[exam_attempts], group_name="scoring", description="Reconstructed midterm score per student-exam.")
def student_exam_scores() -> MaterializeResult:
    pipeline.stage_scores()
    return MaterializeResult(metadata={"student_exams": _rows(pipeline.SCORES)})


@asset(deps=[student_exam_scores], group_name="features", description="Pre-exam practice features (pandas + DuckDB SQL).")
def student_exam_features() -> MaterializeResult:
    pipeline.stage_features()
    return MaterializeResult(metadata={"student_exams": _rows(pipeline.FEATURES)})


@asset(deps=[student_exam_features], group_name="analysis", description="Cohort-controlled correlations, FE models, bootstrap, sensitivity.")
def cohort_analysis() -> None:
    pipeline.stage_analysis()


@asset(deps=[student_exam_features], group_name="analysis", description="Leave-one-semester-out at-risk models, logged to MLflow.")
def at_risk_models() -> MaterializeResult:
    pipeline.stage_models()
    comparison = pd.read_csv(config.TABLES_DIR / "at_risk_model_comparison.csv")
    return MaterializeResult(metadata={"comparison": MetadataValue.md(comparison.round(3).to_markdown(index=False))})


@asset(deps=[cohort_analysis, at_risk_models], group_name="report", description="Figures, headline metrics, Looker Studio CSVs.")
def report() -> None:
    pipeline.stage_report()


defs = Definitions(
    assets=[event_log, exam_attempts, student_exam_scores, student_exam_features, cohort_analysis, at_risk_models, report]
)
