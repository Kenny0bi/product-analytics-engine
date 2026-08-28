"""Funnel analysis engine.

Computes ordered conversion funnels with step-by-step drop-off, temporal
ordering enforcement, and segment comparison.  All heavy lifting is done in
DuckDB SQL to keep memory usage constant regardless of event volume.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
from loguru import logger

from src.analytics.models import FunnelResult, FunnelStep


class FunnelAnalyzer:
    """Compute conversion funnels with step-by-step drop-off analysis.

    A funnel tracks how users progress through an ordered sequence of events.
    Users must complete steps in order: step N requires step N-1 to have
    occurred *earlier* in the same session (or within a configurable day
    window when ``window_days`` is set).

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection to the analytics database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def compute_funnel(
        self,
        steps: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        segment: dict[str, str] | None = None,
        window_days: int | None = None,
    ) -> FunnelResult:
        """Compute funnel conversion for the given ordered event names.

        The query works in three stages:

        1. **step_users** -- For each (user, session) pair, flag which steps
           were reached and record the earliest timestamp per step.
        2. **ordered_steps** -- Enforce temporal ordering: a user only
           "completed" step *k* if they also completed steps 0..k-1 with
           non-decreasing timestamps.
        3. **Aggregation** -- Count distinct users at each step, compute
           step-to-step and overall conversion rates, and extract median
           inter-step times.

        Parameters
        ----------
        steps : list[str]
            Ordered event names defining the funnel (e.g.,
            ``["view_homepage", "view_product", "click_add_to_cart"]``).
        start_date, end_date : str, optional
            ISO date strings to filter the event window.
        segment : dict, optional
            Filter on a user/event dimension, e.g. ``{"device_type": "mobile"}``.
        window_days : int, optional
            If set, allow cross-session funnels within *N* days of the first step.
            Default (None) scopes the funnel to a single session.

        Returns
        -------
        FunnelResult
        """
        if len(steps) < 2:
            raise ValueError("A funnel requires at least 2 steps")

        logger.info("Computing funnel: {}", " -> ".join(steps))

        # Build dynamic SQL for N steps.
        n = len(steps)
        group_col = "e.user_id, e.session_id" if window_days is None else "e.user_id"

        # CTE 1: flag each step reached and record its earliest timestamp.
        reached_cols = []
        time_cols = []
        for i, step in enumerate(steps):
            reached_cols.append(
                f"max(case when e.event_name = '{step}' then 1 else 0 end) as reached_{i}"
            )
            time_cols.append(
                f"min(case when e.event_name = '{step}' then e.timestamp end) as time_{i}"
            )

        where_clauses = ["1=1"]
        if start_date:
            where_clauses.append(f"e.timestamp >= '{start_date}'::timestamp")
        if end_date:
            where_clauses.append(f"e.timestamp <= '{end_date} 23:59:59'::timestamp")
        if segment:
            for col, val in segment.items():
                where_clauses.append(f"e.{col} = '{val}'")
        if window_days is not None:
            # Will be enforced in ordered_steps via time difference.
            pass

        where_sql = " and ".join(where_clauses)

        step_users_sql = f"""
            select
                {group_col},
                {', '.join(reached_cols)},
                {', '.join(time_cols)}
            from events e
            where {where_sql}
            group by {group_col}
        """

        # CTE 2: enforce temporal ordering.
        completed_cols = ["reached_0 as completed_0"]
        for i in range(1, n):
            conditions = []
            for j in range(i + 1):
                conditions.append(f"reached_{j} = 1")
            for j in range(1, i + 1):
                conditions.append(f"time_{j} >= time_{j-1}")
            if window_days is not None:
                conditions.append(
                    f"datediff('day', time_0, time_{i}) <= {window_days}"
                )
            completed_cols.append(
                f"case when {' and '.join(conditions)} then 1 else 0 end as completed_{i}"
            )

        ordered_sql = f"""
            select
                user_id,
                {', '.join(completed_cols)},
                {', '.join(f'time_{i}' for i in range(n))}
            from step_users
        """

        # Final aggregation: one row per step.
        agg_parts = []
        for i in range(n):
            agg_parts.append(
                f"count(distinct case when completed_{i} = 1 then user_id end) as users_{i}"
            )
        # Median time between consecutive steps.
        median_parts = []
        for i in range(n - 1):
            median_parts.append(
                f"median(case when completed_{i+1} = 1 "
                f"then epoch(time_{i+1}) - epoch(time_{i}) end) as median_time_{i}"
            )

        full_sql = f"""
            with step_users as (
                {step_users_sql}
            ),
            ordered_steps as (
                {ordered_sql}
            )
            select
                {', '.join(agg_parts)},
                {', '.join(median_parts) if median_parts else '0 as _dummy'}
            from ordered_steps
        """

        row = self.conn.execute(full_sql).fetchone()

        # Parse results into FunnelStep objects.
        users_at = [row[i] for i in range(n)]
        medians = [row[n + i] for i in range(n - 1)] if n > 1 else []

        funnel_steps: list[FunnelStep] = []
        for i in range(n):
            prev_users = users_at[i - 1] if i > 0 else users_at[0]
            conv_rate = users_at[i] / prev_users if prev_users > 0 else 0.0
            overall_rate = users_at[i] / users_at[0] if users_at[0] > 0 else 0.0
            dropoff = prev_users - users_at[i] if i > 0 else 0
            dropoff_rate = dropoff / prev_users if prev_users > 0 else 0.0

            median_next = None
            if i < len(medians):
                median_next = float(medians[i]) if medians[i] is not None else None

            funnel_steps.append(FunnelStep(
                step_name=steps[i],
                step_index=i,
                users=users_at[i],
                conversion_rate=conv_rate if i > 0 else 1.0,
                overall_conversion_rate=overall_rate,
                dropoff_count=dropoff,
                dropoff_rate=dropoff_rate,
                median_time_to_next=median_next,
            ))

        result = FunnelResult(
            funnel_name=" -> ".join(steps),
            steps=funnel_steps,
            total_entered=users_at[0],
            total_converted=users_at[-1],
            overall_conversion_rate=(
                users_at[-1] / users_at[0] if users_at[0] > 0 else 0.0
            ),
            computed_at=datetime.utcnow(),
        )
        logger.info(
            "Funnel complete: {} entered, {} converted ({:.1%})",
            result.total_entered,
            result.total_converted,
            result.overall_conversion_rate,
        )
        return result

    def compare_funnels(
        self,
        steps: list[str],
        segment_field: str,
        segment_values: list[str],
    ) -> dict[str, FunnelResult]:
        """Compare funnel performance across segment values.

        Runs ``compute_funnel`` once per segment value with the segment
        filter applied, allowing side-by-side comparison of conversion rates
        by device type, UTM source, plan type, etc.

        Parameters
        ----------
        steps : list[str]
            Ordered event names for the funnel.
        segment_field : str
            Column name to segment on (e.g. ``"device_type"``).
        segment_values : list[str]
            Distinct values to compare (e.g. ``["desktop", "mobile", "tablet"]``).

        Returns
        -------
        dict[str, FunnelResult]
            Mapping from segment value to its FunnelResult.
        """
        logger.info("Comparing funnels by {} across {}", segment_field, segment_values)
        results: dict[str, FunnelResult] = {}
        for value in segment_values:
            results[value] = self.compute_funnel(
                steps=steps,
                segment={segment_field: value},
            )
        return results
