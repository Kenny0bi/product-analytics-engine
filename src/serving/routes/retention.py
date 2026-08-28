"""Retention endpoints — cohort retention matrices and average curves.

Wraps RetentionAnalyzer to compute week- or month-based cohort retention
with configurable period count.
"""

from datetime import datetime

import duckdb
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/retention", tags=["retention"])


class RetentionResponse(BaseModel):
    """Cohort retention matrix with rates per cohort per period offset."""

    period_type: str
    cohorts: list[str]
    periods: list[int]
    cohort_sizes: dict[str, int]
    retention_rates: dict[str, list[float]]
    computed_at: datetime


class RetentionCurvePoint(BaseModel):
    """A single point on the average retention curve."""

    period: int
    avg_retention_rate: float
    std_retention_rate: float | None = None


def _get_conn(request: Request) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(request.app.state.db_path, read_only=True)


@router.get("", response_model=RetentionResponse)
async def get_retention(
    request: Request,
    period: str = Query("month", description="Cohort granularity: 'week' or 'month'."),
    num_periods: int = Query(12, ge=1, le=52, description="Number of periods to compute."),
) -> RetentionResponse:
    """Compute and return the cohort retention matrix."""
    conn = _get_conn(request)
    try:
        from src.analytics.retention import RetentionAnalyzer

        analyzer = RetentionAnalyzer(conn)
        matrix = analyzer.compute_retention(period=period, num_periods=num_periods)
        return RetentionResponse(
            period_type=matrix.period_type,
            cohorts=matrix.cohorts,
            periods=matrix.periods,
            cohort_sizes=matrix.cohort_sizes,
            retention_rates=matrix.retention_rates,
            computed_at=matrix.computed_at,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/curve", response_model=list[RetentionCurvePoint])
async def get_retention_curve(
    request: Request,
    period: str = Query("month", description="Cohort granularity: 'week' or 'month'."),
) -> list[RetentionCurvePoint]:
    """Return the average retention curve across all cohorts."""
    conn = _get_conn(request)
    try:
        from src.analytics.retention import RetentionAnalyzer

        analyzer = RetentionAnalyzer(conn)
        curve_df = analyzer.compute_retention_curve(period=period)
        return [
            RetentionCurvePoint(
                period=int(row["period"]),
                avg_retention_rate=float(row["avg_retention_rate"]),
                std_retention_rate=float(row.get("std_retention_rate", 0.0)),
            )
            for _, row in curve_df.iterrows()
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()
