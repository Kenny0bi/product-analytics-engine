"""Session-level analytics.

Computes aggregate session metrics (bounce rate, duration, depth,
conversion) with optional dimensional breakdowns, and per-user engagement
scores combining frequency, depth, breadth, and recency into a single
0-100 index.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd
from loguru import logger

from src.analytics.models import SessionMetrics


class SessionAnalyzer:
    """Compute session-level metrics and per-user engagement scores.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection to the analytics database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------
    # Session metrics
    # ------------------------------------------------------------------

    def compute_session_metrics(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        group_by: str | None = None,
    ) -> SessionMetrics:
        """Aggregate session metrics with optional dimensional breakdown.

        A **bounce** is defined as a session with exactly 1 page view and
        no other event types (``event_count = 1`` and
        ``page_view_count = 1``).

        Parameters
        ----------
        start_date, end_date : str, optional
            ISO date strings to filter the session window.
        group_by : str, optional
            Dimension column for breakdown.  Supported values:
            ``"device_type"``, ``"utm_source"``, ``"day_of_week"``.

        Returns
        -------
        SessionMetrics
        """
        logger.info(
            "Computing session metrics (group_by={})", group_by or "none"
        )

        where_parts = ["1=1"]
        if start_date:
            where_parts.append(f"s.started_at >= '{start_date}'::timestamp")
        if end_date:
            where_parts.append(f"s.started_at <= '{end_date} 23:59:59'::timestamp")
        where_sql = " and ".join(where_parts)

        # Overall metrics.
        overall_sql = f"""
            select
                count(*) as total_sessions,
                avg(s.duration_seconds) as avg_duration,
                median(s.duration_seconds) as median_duration,
                avg(s.page_view_count) as avg_page_depth,
                sum(case when s.event_count = 1 and s.page_view_count = 1
                         then 1 else 0 end) * 1.0 / count(*) as bounce_rate,
                sum(case when s.has_conversion then 1 else 0 end) * 1.0
                    / count(*) as conversion_rate
            from sessions s
            where {where_sql}
        """

        row = self.conn.execute(overall_sql).fetchone()
        total_sessions = int(row[0])
        avg_dur = float(row[1]) if row[1] is not None else 0.0
        med_dur = float(row[2]) if row[2] is not None else 0.0
        avg_depth = float(row[3]) if row[3] is not None else 0.0
        bounce = float(row[4]) if row[4] is not None else 0.0
        conv = float(row[5]) if row[5] is not None else 0.0

        # Dimensional breakdown.
        breakdowns: dict[str, dict[str, float]] | None = None
        if group_by:
            if group_by == "day_of_week":
                dim_expr = "dayname(s.started_at)"
            else:
                dim_expr = f"s.{group_by}"

            breakdown_sql = f"""
                select
                    {dim_expr} as dimension_value,
                    count(*) as total_sessions,
                    avg(s.duration_seconds) as avg_duration,
                    avg(s.page_view_count) as avg_page_depth,
                    sum(case when s.event_count = 1 and s.page_view_count = 1
                             then 1 else 0 end) * 1.0 / count(*) as bounce_rate,
                    sum(case when s.has_conversion then 1 else 0 end) * 1.0
                        / count(*) as conversion_rate
                from sessions s
                where {where_sql}
                group by {dim_expr}
                order by total_sessions desc
            """

            bd_rows = self.conn.execute(breakdown_sql).fetchall()
            breakdowns = {}
            for bd_row in bd_rows:
                dim_val = str(bd_row[0]) if bd_row[0] is not None else "unknown"
                breakdowns[dim_val] = {
                    "total_sessions": float(bd_row[1]),
                    "avg_duration": float(bd_row[2]) if bd_row[2] else 0.0,
                    "avg_page_depth": float(bd_row[3]) if bd_row[3] else 0.0,
                    "bounce_rate": float(bd_row[4]) if bd_row[4] else 0.0,
                    "conversion_rate": float(bd_row[5]) if bd_row[5] else 0.0,
                }

        result = SessionMetrics(
            total_sessions=total_sessions,
            avg_duration_seconds=avg_dur,
            median_duration_seconds=med_dur,
            avg_page_depth=avg_depth,
            bounce_rate=bounce,
            conversion_rate=conv,
            breakdowns=breakdowns,
            computed_at=datetime.utcnow(),
        )

        logger.info(
            "Sessions: {} total, bounce={:.1%}, conversion={:.1%}",
            total_sessions, bounce, conv,
        )
        return result

    # ------------------------------------------------------------------
    # Engagement score
    # ------------------------------------------------------------------

    def compute_engagement_score(self) -> pd.DataFrame:
        """Compute a per-user engagement score from 0 to 100.

        The score is the equally-weighted (25% each) sum of four
        min-max-normalized components:

        1. **Frequency** -- Sessions per week since signup.
        2. **Depth** -- Average events per session.
        3. **Breadth** -- Fraction of distinct feature-use event names
           observed (out of all available feature events).
        4. **Recency** -- Inverse of days since last session, so more
           recent activity scores higher.

        Each component is min-max scaled to [0, 1] across users, then the
        weighted sum is rescaled to [0, 100].

        Returns
        -------
        pd.DataFrame
            Columns: user_id, engagement_score, frequency_component, depth_component,
            breadth_component, recency_component.
        """
        logger.info("Computing per-user engagement scores")

        # Count total available feature_use event names for breadth denominator.
        total_features_row = self.conn.execute("""
            select count(distinct event_name) from events where event_type = 'feature_use'
        """).fetchone()
        total_features = max(total_features_row[0], 1)

        sql = f"""
            with user_metrics as (
                select
                    u.user_id,
                    -- frequency: sessions per week since signup
                    case when datediff('week', u.created_at,
                                       (select max(timestamp) from events)) > 0
                         then count(distinct s.session_id) * 1.0
                              / datediff('week', u.created_at,
                                         (select max(timestamp) from events))
                         else count(distinct s.session_id) * 1.0
                    end as frequency,
                    -- depth: avg events per session
                    coalesce(avg(s.event_count), 0) as depth,
                    -- breadth: unique feature events / total features
                    count(distinct case when e.event_type = 'feature_use'
                                       then e.event_name end) * 1.0
                        / {total_features} as breadth,
                    -- recency: days since last event (lower is better)
                    coalesce(
                        datediff('day', max(e.timestamp),
                                 (select max(timestamp) from events)),
                        365
                    ) as recency_days
                from users u
                left join sessions s on u.user_id = s.user_id
                left join events e on u.user_id = e.user_id
                group by u.user_id, u.created_at
            )
            select
                user_id,
                frequency,
                depth,
                breadth,
                recency_days
            from user_metrics
        """

        df = self.conn.execute(sql).fetchdf()

        # Min-max normalize each component to [0, 1].
        def _minmax(series: pd.Series) -> pd.Series:
            smin, smax = series.min(), series.max()
            if smax == smin:
                return pd.Series(0.5, index=series.index)
            return (series - smin) / (smax - smin)

        df["frequency_norm"] = _minmax(df["frequency"])
        df["depth_norm"] = _minmax(df["depth"])
        df["breadth_norm"] = _minmax(df["breadth"])
        # Invert recency so lower days = higher score.
        df["recency_norm"] = 1.0 - _minmax(df["recency_days"])

        # Composite score: equal weights, scale to 0-100.
        df["engagement_score"] = (
            (df["frequency_norm"] + df["depth_norm"]
             + df["breadth_norm"] + df["recency_norm"])
            / 4.0
            * 100.0
        ).round(2)

        result = df[[
            "user_id", "engagement_score",
            "frequency_norm", "depth_norm", "breadth_norm", "recency_norm",
        ]].rename(columns={
            "frequency_norm": "frequency_component",
            "depth_norm": "depth_component",
            "breadth_norm": "breadth_component",
            "recency_norm": "recency_component",
        })

        logger.info(
            "Engagement scores: mean={:.1f}, median={:.1f}, min={:.1f}, max={:.1f}",
            result["engagement_score"].mean(),
            result["engagement_score"].median(),
            result["engagement_score"].min(),
            result["engagement_score"].max(),
        )
        return result
