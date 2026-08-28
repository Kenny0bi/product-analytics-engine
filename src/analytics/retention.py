"""Cohort retention analysis.

Computes retention matrices for weekly or monthly cohorts.  A user is
"retained" in period *N* if they generated at least one event during the
*N*-th week/month after their signup cohort period.  All computation is
pushed into DuckDB SQL using CTEs and window functions for efficiency.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd
from loguru import logger

from src.analytics.models import RetentionMatrix


class RetentionAnalyzer:
    """Compute cohort retention matrices for weekly and monthly cohorts.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection to the analytics database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def compute_retention(
        self,
        period: str = "month",
        num_periods: int = 12,
        segment: dict[str, str] | None = None,
    ) -> RetentionMatrix:
        """Compute the cohort retention matrix.

        Algorithm
        ---------
        1. Assign each user to a cohort based on ``date_trunc(period, created_at)``.
        2. For every event, compute ``periods_since_signup`` as the integer
           difference (in weeks or months) between the event's truncated
           period and the user's cohort period.
        3. Count distinct active users per (cohort, offset) pair.
        4. Divide by cohort size to get retention rates.

        Parameters
        ----------
        period : str
            ``"week"`` or ``"month"``.
        num_periods : int
            Maximum number of follow-up periods to include (default 12).
        segment : dict, optional
            Filter on a user dimension, e.g. ``{"plan_type": "pro"}``.

        Returns
        -------
        RetentionMatrix
        """
        if period not in ("week", "month"):
            raise ValueError(f"period must be 'week' or 'month', got '{period}'")

        logger.info("Computing {} retention matrix ({} periods)", period, num_periods)

        segment_where = ""
        if segment:
            conditions = [f"u.{col} = '{val}'" for col, val in segment.items()]
            segment_where = "and " + " and ".join(conditions)

        sql = f"""
            with cohorts as (
                select
                    u.user_id,
                    date_trunc('{period}', u.created_at) as cohort_period
                from users u
                where 1=1 {segment_where}
            ),
            user_activity as (
                -- Signing up is itself period-0 activity: every cohort member
                -- is active in their own signup period by definition, which
                -- anchors the matrix at 100% for period 0.
                select
                    c.user_id,
                    c.cohort_period,
                    0 as periods_since_signup
                from cohorts c
                union
                select
                    c.user_id,
                    c.cohort_period,
                    datediff('{period}',
                             c.cohort_period,
                             date_trunc('{period}', e.timestamp)) as periods_since_signup
                from cohorts c
                join events e on c.user_id = e.user_id
                where datediff('{period}',
                               c.cohort_period,
                               date_trunc('{period}', e.timestamp))
                      between 0 and {num_periods}
            ),
            cohort_sizes as (
                select
                    cohort_period,
                    count(distinct user_id) as cohort_size
                from cohorts
                group by cohort_period
            ),
            retention_counts as (
                select
                    ua.cohort_period,
                    ua.periods_since_signup,
                    count(distinct ua.user_id) as active_users
                from user_activity ua
                group by ua.cohort_period, ua.periods_since_signup
            )
            select
                cast(rc.cohort_period as varchar) as cohort_period,
                cs.cohort_size,
                rc.periods_since_signup,
                rc.active_users,
                round(rc.active_users * 100.0 / cs.cohort_size, 2) as retention_rate
            from retention_counts rc
            join cohort_sizes cs on rc.cohort_period = cs.cohort_period
            order by rc.cohort_period, rc.periods_since_signup
        """

        rows = self.conn.execute(sql).fetchall()

        # Parse into RetentionMatrix structures.
        cohort_sizes: dict[str, int] = {}
        retention_rates: dict[str, list[float | None]] = {}
        all_periods = set()

        for cohort_str, size, offset, _active, _rate in rows:
            # Normalize cohort label (strip time component if present).
            label = cohort_str[:10] if period == "month" else cohort_str[:10]
            if period == "month":
                label = label[:7]  # YYYY-MM
            cohort_sizes[label] = size
            if label not in retention_rates:
                retention_rates[label] = []
            all_periods.add(offset)

        # Fill the matrix, ensuring each cohort has a value for every period.
        max_period = max(all_periods) if all_periods else 0
        periods_list = list(range(max_period + 1))

        # Re-parse to fill rates by position.
        rate_map: dict[str, dict[int, float]] = {}
        for cohort_str, _size, offset, _active, rate in rows:
            label = cohort_str[:7] if period == "month" else cohort_str[:10]
            if label not in rate_map:
                rate_map[label] = {}
            rate_map[label][offset] = float(rate)

        filled_rates: dict[str, list[float]] = {}
        for label in sorted(rate_map.keys()):
            filled_rates[label] = [
                rate_map[label].get(p, 0.0) for p in periods_list
            ]

        result = RetentionMatrix(
            period_type=period,
            cohorts=sorted(filled_rates.keys()),
            periods=periods_list,
            cohort_sizes=cohort_sizes,
            retention_rates=filled_rates,
            computed_at=datetime.utcnow(),
        )

        logger.info(
            "Retention matrix: {} cohorts, {} periods",
            len(result.cohorts),
            len(result.periods),
        )
        return result

    def compute_retention_curve(self, period: str = "month") -> pd.DataFrame:
        """Average retention curve across all cohorts.

        Returns a DataFrame with columns ``period`` (offset from signup) and
        ``avg_retention_rate`` (mean rate across cohorts for that offset).
        This smooths out cohort-specific noise and shows the product's
        characteristic retention shape.
        """
        logger.info("Computing average {} retention curve", period)

        sql = f"""
            with cohorts as (
                select
                    u.user_id,
                    date_trunc('{period}', u.created_at) as cohort_period
                from users u
            ),
            user_activity as (
                -- Signup counts as period-0 activity (see compute_retention).
                select
                    c.user_id,
                    c.cohort_period,
                    0 as periods_since_signup
                from cohorts c
                union
                select
                    c.user_id,
                    c.cohort_period,
                    datediff('{period}',
                             c.cohort_period,
                             date_trunc('{period}', e.timestamp)) as periods_since_signup
                from cohorts c
                join events e on c.user_id = e.user_id
                where datediff('{period}',
                               c.cohort_period,
                               date_trunc('{period}', e.timestamp)) >= 0
            ),
            cohort_sizes as (
                select cohort_period, count(distinct user_id) as cohort_size
                from cohorts
                group by cohort_period
            ),
            cohort_retention as (
                select
                    ua.cohort_period,
                    ua.periods_since_signup,
                    count(distinct ua.user_id) * 100.0 /
                        max(cs.cohort_size) as retention_rate
                from user_activity ua
                join cohort_sizes cs on ua.cohort_period = cs.cohort_period
                group by ua.cohort_period, ua.periods_since_signup
            )
            select
                periods_since_signup as period,
                round(avg(retention_rate), 2) as avg_retention_rate
            from cohort_retention
            group by periods_since_signup
            order by periods_since_signup
        """

        df = self.conn.execute(sql).fetchdf()
        logger.info("Retention curve: {} periods", len(df))
        return df
