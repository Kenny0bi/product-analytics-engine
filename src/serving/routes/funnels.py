"""Funnel endpoints — step-by-step conversion analysis and segment comparison.

Wraps FunnelAnalyzer to compute ordered conversion funnels with dropoff rates
and optional segmentation for comparative analysis.
"""

from datetime import datetime

import duckdb
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/funnels", tags=["funnels"])


class FunnelStepResponse(BaseModel):
    """Single step within a computed funnel."""

    step_name: str
    step_index: int
    users: int
    conversion_rate: float
    overall_conversion_rate: float
    dropoff_count: int
    dropoff_rate: float
    median_time_to_next: float | None = None


class FunnelResponse(BaseModel):
    """Complete funnel analysis result."""

    funnel_name: str
    steps: list[FunnelStepResponse]
    total_entered: int
    total_converted: int
    overall_conversion_rate: float
    computed_at: datetime


def _get_conn(request: Request) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(request.app.state.db_path, read_only=True)


def _compute_funnel(
    conn: duckdb.DuckDBPyConnection,
    step_names: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    segment_field: str | None = None,
    segment_value: str | None = None,
) -> FunnelResponse:
    """Build and execute the funnel query, returning structured results."""
    from src.analytics.funnels import FunnelAnalyzer

    analyzer = FunnelAnalyzer(conn)
    segment = None
    if segment_field and segment_value:
        segment = {segment_field: segment_value}
    result = analyzer.compute_funnel(
        steps=step_names,
        start_date=start_date,
        end_date=end_date,
        segment=segment,
    )
    return FunnelResponse(
        funnel_name=result.funnel_name,
        steps=[
            FunnelStepResponse(
                step_name=s.step_name, step_index=s.step_index, users=s.users,
                conversion_rate=s.conversion_rate,
                overall_conversion_rate=s.overall_conversion_rate,
                dropoff_count=s.dropoff_count, dropoff_rate=s.dropoff_rate,
                median_time_to_next=s.median_time_to_next,
            )
            for s in result.steps
        ],
        total_entered=result.total_entered,
        total_converted=result.total_converted,
        overall_conversion_rate=result.overall_conversion_rate,
        computed_at=result.computed_at,
    )


@router.get("", response_model=FunnelResponse)
async def get_funnel(
    request: Request,
    steps: str = Query(..., description="Comma-separated ordered event names."),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    segment_field: str | None = Query(None, description="e.g. 'device_type', 'utm_source'."),
    segment_value: str | None = Query(None),
) -> FunnelResponse:
    """Compute a conversion funnel for the specified step sequence."""
    conn = _get_conn(request)
    try:
        step_names = [s.strip() for s in steps.split(",") if s.strip()]
        if len(step_names) < 2:
            raise HTTPException(status_code=400, detail="At least two funnel steps required.")
        return _compute_funnel(conn, step_names, start_date, end_date, segment_field, segment_value)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/compare", response_model=dict[str, FunnelResponse])
async def compare_funnels(
    request: Request,
    steps: str = Query(..., description="Comma-separated ordered event names."),
    segment_field: str = Query(..., description="Dimension to compare, e.g. 'device_type'."),
) -> dict[str, FunnelResponse]:
    """Compare funnel performance across all values of a segment dimension."""
    conn = _get_conn(request)
    try:
        from src.analytics.funnels import FunnelAnalyzer

        step_names = [s.strip() for s in steps.split(",") if s.strip()]
        analyzer = FunnelAnalyzer(conn)
        results = analyzer.compare_funnels(step_names, segment_field, segment_values=[])
        return {
            key: FunnelResponse(
                funnel_name=val.funnel_name,
                steps=[
                    FunnelStepResponse(
                        step_name=s.step_name, step_index=s.step_index, users=s.users,
                        conversion_rate=s.conversion_rate,
                        overall_conversion_rate=s.overall_conversion_rate,
                        dropoff_count=s.dropoff_count, dropoff_rate=s.dropoff_rate,
                        median_time_to_next=s.median_time_to_next,
                    )
                    for s in val.steps
                ],
                total_entered=val.total_entered,
                total_converted=val.total_converted,
                overall_conversion_rate=val.overall_conversion_rate,
                computed_at=val.computed_at,
            )
            for key, val in results.items()
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()
