"""Forecasting and anomaly detection endpoints.

Wraps MetricForecaster for Holt-Winters time series forecasting with prediction
intervals, and AnomalyDetector for Isolation Forest and control chart-based
anomaly flagging.
"""

from datetime import date, datetime

import duckdb
from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["forecasting"])


class ForecastPoint(BaseModel):
    """A single forecasted value with prediction intervals."""

    date: date
    predicted_value: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float


class ForecastResponse(BaseModel):
    """Complete forecast result with historical context."""

    metric_name: str
    periods_ahead: int
    forecast: list[ForecastPoint]
    computed_at: datetime


class AnomalyResponse(BaseModel):
    """A detected anomaly in a metric time series."""

    date: date
    metric_value: float
    expected_value: float
    deviation: float
    upper_control_limit: float
    lower_control_limit: float
    severity: str


def _get_conn(request: Request) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(request.app.state.db_path, read_only=True)


@router.get("/forecast/{metric_name}", response_model=ForecastResponse)
async def get_forecast(
    request: Request,
    metric_name: str = Path(..., description="Metric to forecast, e.g. 'dau', 'revenue'."),
    periods_ahead: int = Query(30, ge=1, le=365, description="Number of days to forecast."),
) -> ForecastResponse:
    """Forecast a metric using Holt-Winters exponential smoothing."""
    conn = _get_conn(request)
    try:
        from src.statistics.forecasting import MetricForecaster

        forecaster = MetricForecaster(conn)
        result = forecaster.forecast_metric(metric_name=metric_name, periods_ahead=periods_ahead)

        forecast_points = [
            ForecastPoint(
                date=d, predicted_value=v,
                lower_80=l80, upper_80=u80,
                lower_95=l95, upper_95=u95,
            )
            for d, v, l80, u80, l95, u95 in zip(
                result.forecast_dates, result.forecast_values,
                result.lower_80, result.upper_80,
                result.lower_95, result.upper_95,
            )
        ]
        return ForecastResponse(
            metric_name=metric_name,
            periods_ahead=periods_ahead,
            forecast=forecast_points,
            computed_at=datetime.utcnow(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/anomalies/{metric_name}", response_model=list[AnomalyResponse])
async def get_anomalies(
    request: Request,
    metric_name: str = Path(..., description="Metric to analyze for anomalies."),
    lookback_days: int = Query(90, ge=7, le=365, description="Number of days to analyze."),
) -> list[AnomalyResponse]:
    """Detect anomalies in a metric's recent history using Isolation Forest and control charts."""
    conn = _get_conn(request)
    try:
        from src.statistics.anomaly_detection import AnomalyDetector

        detector = AnomalyDetector(conn)
        anomalies = detector.detect_metric_anomalies(
            metric_name=metric_name, lookback_days=lookback_days,
        )
        # Severity from the rolling z-score: control-chart breaches are the
        # most serious, then graded by how many sigmas the day sits from its
        # rolling mean.
        def _severity(a) -> str:
            if a.is_control_chart_anomaly or abs(a.z_score) >= 4:
                return "high"
            if abs(a.z_score) >= 2.5:
                return "medium"
            return "low"

        return [
            AnomalyResponse(
                date=date.fromisoformat(a.metric_date),
                metric_value=a.metric_value,
                expected_value=a.expected_value,
                deviation=round(a.metric_value - a.expected_value, 4),
                upper_control_limit=a.ucl,
                lower_control_limit=a.lcl,
                severity=_severity(a),
            )
            for a in anomalies
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()
