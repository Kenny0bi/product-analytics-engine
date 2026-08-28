"""Metrics endpoints — daily time series and current summary snapshots.

Provides access to pre-computed daily_metrics rows with optional dimensional
breakdowns, and a summary endpoint that returns the latest DAU, WAU, MAU,
revenue, conversion rate, and average session duration.
"""

from datetime import date, datetime

import duckdb
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


class MetricDataPoint(BaseModel):
    """A single metric observation for one date and optional dimension slice."""

    metric_date: date
    metric_name: str
    metric_value: float
    dimension_name: str | None = None
    dimension_value: str | None = None


class MetricsSummary(BaseModel):
    """Current-state KPI snapshot across core product health metrics."""

    dau: int
    wau: int
    mau: int
    total_revenue: float
    conversion_rate: float
    avg_session_duration_seconds: float
    computed_at: datetime


def _get_conn(request: Request) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(request.app.state.db_path, read_only=True)


@router.get("/daily", response_model=list[MetricDataPoint])
async def get_daily_metrics(
    request: Request,
    metric_name: str = Query(..., description="Metric to retrieve, e.g. 'dau', 'revenue'."),
    start_date: str | None = Query(None, description="ISO date lower bound (inclusive)."),
    end_date: str | None = Query(None, description="ISO date upper bound (inclusive)."),
    dimension: str | None = Query(None, description="Dimension to filter by, e.g. 'device_type'."),
    dimension_value: str | None = Query(None, description="Specific dimension value."),
) -> list[MetricDataPoint]:
    """Return daily metric values with optional date range and dimension filters.

    The daily_metrics table stores one row per (date, metric, dimension_name,
    dimension_value). Passing *dimension* without *dimension_value* returns all
    values for that dimension; passing both narrows to a single slice.
    """
    conn = _get_conn(request)
    try:
        conditions = ["metric_name = ?"]
        params: list[str] = [metric_name]

        if start_date:
            conditions.append("metric_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("metric_date <= ?")
            params.append(end_date)
        if dimension:
            conditions.append("dimension_name = ?")
            params.append(dimension)
        if dimension_value:
            conditions.append("dimension_value = ?")
            params.append(dimension_value)

        where_clause = " and ".join(conditions)
        query = f"""
            select metric_date, metric_name, metric_value, dimension_name, dimension_value
            from daily_metrics
            where {where_clause}
            order by metric_date
        """
        rows = conn.execute(query, params).fetchall()
        return [
            MetricDataPoint(
                metric_date=r[0], metric_name=r[1], metric_value=float(r[2]),
                dimension_name=r[3], dimension_value=r[4],
            )
            for r in rows
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/summary", response_model=MetricsSummary)
async def get_metrics_summary(request: Request) -> MetricsSummary:
    """Return a current-state KPI snapshot.

    All figures anchor on the last *complete* day of data (the day before
    the newest event date): the newest date is usually still filling and
    would report a misleadingly tiny DAU. WAU and MAU use trailing 7-day
    and 30-day windows from that anchor; revenue, conversion rate, and
    session duration are trailing 30-day figures.
    """
    conn = _get_conn(request)
    try:
        max_date_row = conn.execute(
            "select max(metric_date) - 1 from daily_metrics"
        ).fetchone()
        ref_date = max_date_row[0] if max_date_row and max_date_row[0] else date.today()

        def _scalar(query: str, params: list | None = None) -> float:
            row = conn.execute(query, params or []).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0

        dau = _scalar(
            "select metric_value from daily_metrics where metric_name = 'dau' and metric_date = ? and dimension_name = 'overall'",
            [ref_date],
        )
        wau = _scalar(
            "select count(distinct e.user_id) from events e where cast(e.timestamp as date) > ? - interval '7 days' and cast(e.timestamp as date) <= ?",
            [ref_date, ref_date],
        )
        mau = _scalar(
            "select count(distinct e.user_id) from events e where cast(e.timestamp as date) > ? - interval '30 days' and cast(e.timestamp as date) <= ?",
            [ref_date, ref_date],
        )
        revenue = _scalar(
            "select coalesce(sum(metric_value), 0) from daily_metrics where metric_name = 'revenue' and metric_date > ? - interval '30 days' and metric_date <= ? and dimension_name = 'overall'",
            [ref_date, ref_date],
        )
        conversion_rate = _scalar(
            "select avg(metric_value) from daily_metrics where metric_name = 'conversion_rate' and metric_date > ? - interval '30 days' and metric_date <= ? and dimension_name = 'overall'",
            [ref_date, ref_date],
        )
        avg_duration = _scalar(
            "select avg(metric_value) from daily_metrics where metric_name = 'avg_session_duration' and metric_date > ? - interval '30 days' and metric_date <= ? and dimension_name = 'overall'",
            [ref_date, ref_date],
        )

        return MetricsSummary(
            dau=int(dau),
            wau=int(wau),
            mau=int(mau),
            total_revenue=revenue,
            conversion_rate=conversion_rate,
            avg_session_duration_seconds=avg_duration,
            computed_at=datetime.utcnow(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()
