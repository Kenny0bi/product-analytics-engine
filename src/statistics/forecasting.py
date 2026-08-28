"""
Metric forecasting using Holt-Winters exponential smoothing.

The forecaster queries the ``daily_metrics`` table for a given metric, fits a
Holt-Winters model via ``statsmodels.tsa.holtwinters.ExponentialSmoothing``,
and produces point forecasts with 80 % and 95 % prediction intervals.

Model selection
---------------
* If the series spans at least two full seasonal cycles (14 days for daily
  data with weekly seasonality), the additive Holt-Winters model is used:
  ``trend='add', seasonal='add', seasonal_periods=7``.
* For shorter series the seasonal component is dropped:
  ``trend='add', seasonal=None``.
* If the series has fewer than 4 observations, a naive last-value forecast
  is returned as a safe fallback.

Prediction intervals are computed from the model's residual standard error
assuming normally distributed forecast errors:
  PI = forecast +/- z * sigma_residuals * sqrt(h)
where h is the forecast horizon step.

References
----------
- Hyndman, R. & Athanasopoulos, G. *Forecasting: Principles and Practice*,
  3rd ed., Ch. 8 (Exponential Smoothing).
- statsmodels ExponentialSmoothing documentation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class ForecastResult(BaseModel):
    """Forecast output with point predictions and prediction intervals."""

    metric_name: str
    frequency: str  # "D" or "W"
    periods_ahead: int
    forecast_values: list[float]
    forecast_dates: list[str]
    lower_80: list[float]
    upper_80: list[float]
    lower_95: list[float]
    upper_95: list[float]
    model_type: str  # "holt-winters" | "holt" | "naive"
    residual_std: float
    computed_at: datetime


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class MetricForecaster:
    """Forecast product metrics stored in the ``daily_metrics`` table.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active connection to the analytics database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def _fetch_metric_series(
        self, metric_name: str, frequency: str
    ) -> pd.Series:
        """Query daily_metrics and return a time-indexed Series."""
        df = self.conn.execute(
            """
            select metric_date, metric_value
            from daily_metrics
            where metric_name = ?
              and dimension_name = 'overall'
              -- Exclude the newest (structurally incomplete) day: fitting
              -- on a partial day drags the model level toward zero at the
              -- series edge.
              and metric_date < (
                  select max(metric_date) from daily_metrics
                  where metric_name = ? and dimension_name = 'overall'
              )
            order by metric_date
            """,
            [metric_name, metric_name],
        ).fetchdf()

        if df.empty:
            raise ValueError(f"No data found for metric '{metric_name}'")

        df["metric_date"] = pd.to_datetime(df["metric_date"])
        df = df.set_index("metric_date").sort_index()
        series = df["metric_value"].astype(float)

        # Resample if weekly frequency requested
        if frequency == "W":
            series = series.resample("W").mean().dropna()

        # Ensure a regular frequency index for statsmodels
        series = series.asfreq(frequency, method="ffill")

        return series

    def forecast_metric(
        self,
        metric_name: str,
        periods_ahead: int = 30,
        frequency: str = "D",
    ) -> ForecastResult:
        """Produce a point forecast with prediction intervals.

        Parameters
        ----------
        metric_name : str
            Name of the metric in ``daily_metrics`` (e.g. ``"dau"``).
        periods_ahead : int
            Number of future periods to forecast (default 30).
        frequency : str
            ``"D"`` for daily (default), ``"W"`` for weekly.

        Returns
        -------
        ForecastResult
        """
        series = self._fetch_metric_series(metric_name, frequency)
        n = len(series)

        seasonal_period = 7 if frequency == "D" else 4  # weekly or monthly cycle

        # ------------------------------------------------------------------
        # Model selection based on series length
        # ------------------------------------------------------------------
        if n < 4:
            # Naive fallback: repeat last value
            last_val = float(series.iloc[-1])
            forecast_vals = [last_val] * periods_ahead
            residual_std = float(series.std()) if n > 1 else 0.0
            model_type = "naive"
            logger.warning(
                "Series '{}' has only {} points; using naive forecast",
                metric_name,
                n,
            )
        elif n < 2 * seasonal_period:
            # Not enough data for seasonal model: Holt's linear trend only
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=None,
            ).fit(optimized=True)
            forecast_vals = model.forecast(periods_ahead).tolist()
            residual_std = float(np.std(model.resid.dropna()))
            model_type = "holt"
            logger.info(
                "Fitted Holt linear trend for '{}' (n={})", metric_name, n
            )
        else:
            # Full Holt-Winters with additive trend and seasonality
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_period,
            ).fit(optimized=True)
            forecast_vals = model.forecast(periods_ahead).tolist()
            residual_std = float(np.std(model.resid.dropna()))
            model_type = "holt-winters"
            logger.info(
                "Fitted Holt-Winters for '{}' (n={}, seasonal_periods={})",
                metric_name,
                n,
                seasonal_period,
            )

        # ------------------------------------------------------------------
        # Prediction intervals: forecast +/- z * sigma * sqrt(h)
        # ------------------------------------------------------------------
        z_80 = 1.2816  # norm.ppf(0.90)
        z_95 = 1.9600  # norm.ppf(0.975)

        lower_80, upper_80 = [], []
        lower_95, upper_95 = [], []
        for h in range(1, periods_ahead + 1):
            spread = residual_std * np.sqrt(h)
            fc = forecast_vals[h - 1]
            lower_80.append(round(fc - z_80 * spread, 4))
            upper_80.append(round(fc + z_80 * spread, 4))
            lower_95.append(round(fc - z_95 * spread, 4))
            upper_95.append(round(fc + z_95 * spread, 4))

        # Build forecast date index
        last_date = series.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(frequency),
            periods=periods_ahead,
            freq=frequency,
        )
        forecast_dates = [d.strftime("%Y-%m-%d") for d in forecast_index]

        return ForecastResult(
            metric_name=metric_name,
            frequency=frequency,
            periods_ahead=periods_ahead,
            forecast_values=[round(v, 4) for v in forecast_vals],
            forecast_dates=forecast_dates,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            model_type=model_type,
            residual_std=round(residual_std, 4),
            computed_at=datetime.now(timezone.utc),
        )
