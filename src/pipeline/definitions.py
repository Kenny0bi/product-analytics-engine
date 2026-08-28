"""
Dagster repository definitions — jobs, schedules, and the top-level Definitions object.

Jobs
----
* ``daily_metrics_job`` — materializes all daily-partitioned metric assets and
  the ``daily_metrics_materialized`` sink.  Scheduled at 02:00 UTC every day.
* ``weekly_analytics_job`` — materializes ``retention_matrix`` and
  ``rfm_segments``.  Scheduled at 03:00 UTC every Monday.

The ``experiment_results`` asset is included in the daily job so that running
experiments are re-analyzed each day alongside standard metrics.
"""

from __future__ import annotations

from dagster import (
    Definitions,
    ScheduleDefinition,
    define_asset_job,
)

from src.pipeline.assets import (
    daily_active_users,
    daily_conversion_rate,
    daily_metrics_materialized,
    daily_revenue,
    daily_session_metrics,
    daily_signups,
    experiment_results,
    retention_matrix,
    rfm_segments,
)
from src.pipeline.resources import DuckDBResource

# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

daily_metrics_job = define_asset_job(
    name="daily_metrics_job",
    selection=[
        daily_active_users,
        daily_revenue,
        daily_signups,
        daily_conversion_rate,
        daily_session_metrics,
        daily_metrics_materialized,
        experiment_results,
    ],
    description="Compute all daily metrics and upsert into daily_metrics table.",
)

weekly_analytics_job = define_asset_job(
    name="weekly_analytics_job",
    selection=[
        retention_matrix,
        rfm_segments,
    ],
    description="Recompute retention matrix and RFM segments.",
)

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

daily_schedule = ScheduleDefinition(
    job=daily_metrics_job,
    cron_schedule="0 2 * * *",  # 02:00 UTC daily
    execution_timezone="UTC",
)

weekly_schedule = ScheduleDefinition(
    job=weekly_analytics_job,
    cron_schedule="0 3 * * 1",  # 03:00 UTC every Monday
    execution_timezone="UTC",
)

# ---------------------------------------------------------------------------
# Definitions (top-level entry point for `dagster dev -m src.pipeline.definitions`)
# ---------------------------------------------------------------------------

defs = Definitions(
    assets=[
        daily_active_users,
        daily_revenue,
        daily_signups,
        daily_conversion_rate,
        daily_session_metrics,
        daily_metrics_materialized,
        retention_matrix,
        rfm_segments,
        experiment_results,
    ],
    schedules=[daily_schedule, weekly_schedule],
    jobs=[daily_metrics_job, weekly_analytics_job],
    resources={
        "duckdb": DuckDBResource(db_path="data/analytics.duckdb"),
    },
)
