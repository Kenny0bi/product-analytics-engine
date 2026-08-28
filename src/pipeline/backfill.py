"""One-shot backfill of the daily_metrics table.

The Dagster assets in ``src.pipeline.assets`` compute one partition (one day)
per run, which is the right shape for scheduled production runs. For local
development and the CLI ``analyze`` command, recomputing the entire history
partition-by-partition would mean hundreds of separate runs, so this module
materializes the same metrics for every day in a single set-based SQL pass.

The metric definitions here must stay in agreement with the per-partition
assets; both write the same (metric_date, metric_name, 'overall', 'overall')
rows and are idempotent via delete-then-insert.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger

_OVERALL_METRICS = ["dau", "revenue", "signups", "conversion_rate",
                    "avg_session_duration", "bounce_rate"]


def backfill_daily_metrics(conn: duckdb.DuckDBPyConnection) -> int:
    """Recompute daily_metrics for the full event history.

    Deletes existing overall-dimension rows for the metrics it owns, then
    inserts fresh values computed from the events, users, and sessions
    tables. Returns the number of rows inserted.

    Metrics
    -------
    - dau: distinct users with any event that day
    - revenue: sum of purchase revenue that day
    - signups: users created that day
    - conversion_rate: share of that day's sessions with a funnel conversion
    - avg_session_duration: mean session duration in seconds
    - bounce_rate: share of sessions with exactly one page view and no
      other events
    """
    logger.info("Backfilling daily_metrics for full history")

    placeholders = ", ".join(f"'{m}'" for m in _OVERALL_METRICS)
    conn.execute(
        f"delete from daily_metrics where metric_name in ({placeholders}) "
        "and dimension_name = 'overall'"
    )

    conn.execute("""
        insert into daily_metrics
        select cast(e.timestamp as date) as metric_date,
               'dau' as metric_name,
               count(distinct e.user_id) as metric_value,
               'overall', 'overall'
        from events e
        group by 1

        union all

        select cast(e.timestamp as date),
               'revenue',
               coalesce(sum(e.revenue), 0),
               'overall', 'overall'
        from events e
        group by 1

        union all

        select cast(u.created_at as date),
               'signups',
               count(*),
               'overall', 'overall'
        from users u
        group by 1

        union all

        select cast(s.started_at as date),
               'conversion_rate',
               avg(case when s.has_conversion then 1.0 else 0.0 end),
               'overall', 'overall'
        from sessions s
        group by 1

        union all

        select cast(s.started_at as date),
               'avg_session_duration',
               avg(s.duration_seconds),
               'overall', 'overall'
        from sessions s
        group by 1

        union all

        select cast(s.started_at as date),
               'bounce_rate',
               avg(case when s.event_count = 1 and s.page_view_count = 1
                        then 1.0 else 0.0 end),
               'overall', 'overall'
        from sessions s
        group by 1
    """)

    count = conn.execute(
        f"select count(*) from daily_metrics where metric_name in ({placeholders})"
    ).fetchone()[0]
    logger.info("daily_metrics backfilled: {} rows across {} metrics",
                count, len(_OVERALL_METRICS))
    return int(count)


def refresh_materialized_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Rebuild the pre-aggregated tables in sql/materialized_views.sql.

    DuckDB has no native materialized views, so these are plain tables
    rebuilt with drop-and-create. Idempotent by construction.
    """
    sql_path = (Path(__file__).resolve().parent.parent.parent
                / "sql" / "materialized_views.sql")
    import re
    sql_text = re.sub(r"--[^\n]*", "", sql_path.read_text())
    for statement in sql_text.split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)
    logger.info("Materialized views refreshed from {}", sql_path.name)
