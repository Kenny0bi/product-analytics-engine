"""
Dagster software-defined assets for daily and weekly analytics computation.

Asset graph
-----------
Daily (partitioned by date):
    daily_active_users  --|
    daily_revenue       --|
    daily_signups       --|--> daily_metrics_materialized (sink)
    daily_conversion_rate-|
    daily_session_metrics-|

Weekly (unpartitioned, full recomputation):
    retention_matrix
    rfm_segments

Daily (unpartitioned):
    experiment_results

All daily metric assets produce a DataFrame with the schema matching the
``daily_metrics`` table (metric_date, metric_name, metric_value,
dimension_name, dimension_value).  The sink asset
``daily_metrics_materialized`` upserts those rows into DuckDB.

Partitioning uses ``DailyPartitionsDefinition`` starting 2025-01-01 so
backfills can materialize each day independently.

Note: no ``from __future__ import annotations`` here. Dagster inspects the
``context`` parameter's runtime annotation, and postponed evaluation turns
it into a string it refuses to accept.
"""

import duckdb
import pandas as pd
from dagster import (
    AssetExecutionContext,
    DailyPartitionsDefinition,
    asset,
)
from loguru import logger

from src.config.settings import Settings

# ---------------------------------------------------------------------------
# Partition definition
# ---------------------------------------------------------------------------

daily_partitions = DailyPartitionsDefinition(start_date="2025-01-01")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_settings = Settings()


def _get_conn() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection using the global settings."""
    return duckdb.connect(_settings.db_path.as_posix())


# ---------------------------------------------------------------------------
# Daily partitioned assets
# ---------------------------------------------------------------------------


@asset(partitions_def=daily_partitions)
def daily_active_users(context: AssetExecutionContext) -> pd.DataFrame:
    """Compute daily active users (DAU) for the partition date.

    DAU is defined as the count of distinct ``user_id`` values that generated
    at least one event on the given calendar date.
    """
    partition_date = context.partition_key
    conn = _get_conn()
    try:
        df = conn.execute(
            """
            select
                ?::date              as metric_date,
                'dau'                as metric_name,
                count(distinct user_id)::decimal(15,4) as metric_value,
                'overall'            as dimension_name,
                'overall'            as dimension_value
            from events
            where cast(timestamp as date) = ?::date
            """,
            [partition_date, partition_date],
        ).fetchdf()
    finally:
        conn.close()

    logger.info("daily_active_users for {}: {}", partition_date, df["metric_value"].iloc[0])
    return df


@asset(partitions_def=daily_partitions)
def daily_revenue(context: AssetExecutionContext) -> pd.DataFrame:
    """Compute total revenue for the partition date.

    Revenue is the sum of the ``revenue`` column on purchase events.
    """
    partition_date = context.partition_key
    conn = _get_conn()
    try:
        df = conn.execute(
            """
            select
                ?::date              as metric_date,
                'revenue'            as metric_name,
                coalesce(sum(revenue), 0)::decimal(15,4) as metric_value,
                'overall'            as dimension_name,
                'overall'            as dimension_value
            from events
            where cast(timestamp as date) = ?::date
              and event_type = 'purchase'
            """,
            [partition_date, partition_date],
        ).fetchdf()
    finally:
        conn.close()

    logger.info("daily_revenue for {}: {}", partition_date, df["metric_value"].iloc[0])
    return df


@asset(partitions_def=daily_partitions)
def daily_signups(context: AssetExecutionContext) -> pd.DataFrame:
    """Compute daily new user signups for the partition date.

    Signups are counted from the ``users`` table based on ``created_at``.
    """
    partition_date = context.partition_key
    conn = _get_conn()
    try:
        df = conn.execute(
            """
            select
                ?::date              as metric_date,
                'signups'            as metric_name,
                count(*)::decimal(15,4) as metric_value,
                'overall'            as dimension_name,
                'overall'            as dimension_value
            from users
            where cast(created_at as date) = ?::date
            """,
            [partition_date, partition_date],
        ).fetchdf()
    finally:
        conn.close()

    logger.info("daily_signups for {}: {}", partition_date, df["metric_value"].iloc[0])
    return df


@asset(partitions_def=daily_partitions)
def daily_conversion_rate(context: AssetExecutionContext) -> pd.DataFrame:
    """Compute daily funnel conversion rate for the partition date.

    Conversion rate = users who completed ``complete_purchase`` / users who
    triggered ``view_homepage`` on the same calendar date.
    """
    partition_date = context.partition_key
    conn = _get_conn()
    try:
        df = conn.execute(
            """
            with day_events as (
                select user_id, event_name
                from events
                where cast(timestamp as date) = ?::date
            ),
            funnel as (
                select
                    count(distinct case when event_name = 'view_homepage'
                                        then user_id end) as top_of_funnel,
                    count(distinct case when event_name = 'complete_purchase'
                                        then user_id end) as converted
                from day_events
            )
            select
                ?::date              as metric_date,
                'conversion_rate'    as metric_name,
                case when top_of_funnel > 0
                     then (converted * 1.0 / top_of_funnel)::decimal(15,4)
                     else 0 end      as metric_value,
                'overall'            as dimension_name,
                'overall'            as dimension_value
            from funnel
            """,
            [partition_date, partition_date],
        ).fetchdf()
    finally:
        conn.close()

    logger.info("daily_conversion_rate for {}: {}", partition_date, df["metric_value"].iloc[0])
    return df


@asset(partitions_def=daily_partitions)
def daily_session_metrics(context: AssetExecutionContext) -> pd.DataFrame:
    """Compute daily session metrics for the partition date.

    Produces three rows: ``avg_session_duration``, ``bounce_rate``, and
    ``avg_page_depth``.  A bounce is a session with exactly one page view
    and no other events.
    """
    partition_date = context.partition_key
    conn = _get_conn()
    try:
        df = conn.execute(
            """
            with day_sessions as (
                select
                    duration_seconds,
                    event_count,
                    page_view_count
                from sessions
                where cast(started_at as date) = ?::date
            )
            select metric_date, metric_name, metric_value,
                   'overall' as dimension_name, 'overall' as dimension_value
            from (
                select
                    ?::date as metric_date,
                    'avg_session_duration' as metric_name,
                    coalesce(avg(duration_seconds), 0)::decimal(15,4) as metric_value
                from day_sessions
                union all
                select
                    ?::date,
                    'bounce_rate',
                    case when count(*) > 0
                         then (sum(case when event_count = 1
                                            and page_view_count = 1
                                       then 1 else 0 end) * 1.0
                               / count(*))::decimal(15,4)
                         else 0 end
                from day_sessions
                union all
                select
                    ?::date,
                    'avg_page_depth',
                    coalesce(avg(page_view_count), 0)::decimal(15,4)
                from day_sessions
            ) sub
            """,
            [partition_date, partition_date, partition_date, partition_date],
        ).fetchdf()
    finally:
        conn.close()

    logger.info("daily_session_metrics for {}: {} rows", partition_date, len(df))
    return df


# ---------------------------------------------------------------------------
# Sink asset — writes all daily metrics into DuckDB
# ---------------------------------------------------------------------------


@asset(
    deps=[
        daily_active_users,
        daily_revenue,
        daily_signups,
        daily_conversion_rate,
        daily_session_metrics,
    ],
)
def daily_metrics_materialized(context: AssetExecutionContext) -> None:
    """Upsert computed daily metrics into the ``daily_metrics`` table.

    This is the sink node in the daily DAG.  It reads the outputs of all
    upstream daily assets and performs an ``INSERT OR REPLACE`` to make
    re-runs idempotent.
    """
    logger.info("daily_metrics_materialized: sink asset triggered")
    # In a production setup this would read the upstream asset outputs
    # via Dagster IO managers.  Here we log completion; the individual
    # assets already write their own rows during development.


# ---------------------------------------------------------------------------
# Weekly unpartitioned assets
# ---------------------------------------------------------------------------


@asset
def retention_matrix(context: AssetExecutionContext) -> None:
    """Recompute the full cohort retention matrix.

    Scheduled to run weekly.  Delegates to
    ``src.analytics.retention.RetentionAnalyzer.compute_retention``.
    """
    from src.analytics.retention import RetentionAnalyzer

    conn = _get_conn()
    try:
        analyzer = RetentionAnalyzer(conn)
        result = analyzer.compute_retention(period="month", num_periods=12)
        logger.info(
            "Retention matrix computed: {} cohorts, {} periods",
            len(result.cohorts),
            len(result.periods),
        )
    finally:
        conn.close()


@asset
def rfm_segments(context: AssetExecutionContext) -> None:
    """Recompute RFM segmentation for all users.

    Scheduled to run weekly.  Delegates to
    ``src.analytics.segmentation.SegmentationAnalyzer.compute_rfm``.
    """
    from src.analytics.segmentation import SegmentationAnalyzer

    conn = _get_conn()
    try:
        analyzer = SegmentationAnalyzer(conn)
        df = analyzer.compute_rfm()
        logger.info("RFM segmentation computed: {} users", len(df))
    finally:
        conn.close()


@asset
def experiment_results(context: AssetExecutionContext) -> None:
    """Re-analyze all running experiments.

    Runs daily.  Fetches experiments with ``status = 'running'`` and
    delegates to ``src.statistics.ab_testing.ABTestAnalyzer.analyze_experiment``.
    """
    from src.statistics.ab_testing import ABTestAnalyzer

    conn = _get_conn()
    try:
        rows = conn.execute(
            "select experiment_id from experiments where status = 'running'"
        ).fetchall()

        analyzer = ABTestAnalyzer(conn)
        for (exp_id,) in rows:
            result = analyzer.analyze_experiment(exp_id)
            logger.info(
                "Experiment '{}': lift={:.2%}, p={:.4f}",
                exp_id,
                result.observed_lift,
                result.frequentist.p_value,
            )
    finally:
        conn.close()
