"""
Anomaly detection for product metrics using Isolation Forest and control charts.

Two complementary methods are applied:

1. **Isolation Forest** (sklearn) — an unsupervised ensemble that isolates
   anomalies by random recursive partitioning.  Observations that require
   fewer splits to isolate receive higher anomaly scores.  Features per day:

   * ``metric_value`` — the raw metric.
   * ``day_of_week`` — integer 0-6 encoding weekly seasonality.
   * ``rolling_mean_7d`` — 7-day trailing mean.
   * ``rolling_std_7d`` — 7-day trailing standard deviation.
   * ``pct_change`` — day-over-day percentage change.
   * ``z_score`` — deviation from the rolling mean in standard-deviation units.

2. **Control charts** (Shewhart-style) — flag any value outside the
   mean +/- 3 sigma band computed over a rolling window.  UCL (upper
   control limit) and LCL (lower control limit) are returned for
   visualization.

References
----------
- Liu, Ting, Zhou. "Isolation Forest" (ICDM 2008).
- Montgomery, D. *Introduction to Statistical Quality Control*, 7th ed.,
  Ch. 5 (Shewhart charts).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class AnomalyResult(BaseModel):
    """A single detected anomaly with context."""

    metric_date: str
    metric_value: float
    expected_value: float  # rolling mean
    z_score: float
    anomaly_score: float  # Isolation Forest decision_function value
    is_control_chart_anomaly: bool  # outside UCL / LCL
    ucl: float
    lcl: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Detect anomalies in daily product metrics.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active connection to the analytics database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def _fetch_metric_series(
        self, metric_name: str, lookback_days: int
    ) -> pd.DataFrame:
        """Return a DataFrame with metric_date and metric_value columns."""
        df = self.conn.execute(
            """
            select metric_date, metric_value
            from daily_metrics
            where metric_name = ?
              and dimension_name = 'overall'
              -- Exclude the newest (structurally incomplete) day, which
              -- would otherwise always be flagged as a false anomaly.
              and metric_date < (
                  select max(metric_date) from daily_metrics
                  where metric_name = ? and dimension_name = 'overall'
              )
            order by metric_date desc
            limit ?
            """,
            [metric_name, metric_name, lookback_days],
        ).fetchdf()

        if df.empty:
            raise ValueError(f"No data found for metric '{metric_name}'")

        df["metric_date"] = pd.to_datetime(df["metric_date"])
        df = df.sort_values("metric_date").reset_index(drop=True)
        df["metric_value"] = df["metric_value"].astype(float)
        return df

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered features for the anomaly model.

        Adds rolling statistics, percentage change, and z-score columns
        to the input DataFrame (in place).  Rows with insufficient history
        for the rolling window are filled forward or set to zero.
        """
        df = df.copy()
        df["day_of_week"] = df["metric_date"].dt.dayofweek
        df["rolling_mean_7d"] = (
            df["metric_value"].rolling(window=7, min_periods=1).mean()
        )
        df["rolling_std_7d"] = (
            df["metric_value"].rolling(window=7, min_periods=1).std().fillna(0)
        )
        df["pct_change"] = df["metric_value"].pct_change().fillna(0)

        # Z-score relative to rolling statistics
        df["z_score"] = np.where(
            df["rolling_std_7d"] > 0,
            (df["metric_value"] - df["rolling_mean_7d"]) / df["rolling_std_7d"],
            0.0,
        )
        return df

    def detect_metric_anomalies(
        self,
        metric_name: str,
        contamination: float = 0.05,
        lookback_days: int = 90,
    ) -> list[AnomalyResult]:
        """Detect anomalies in a metric's recent history.

        Parameters
        ----------
        metric_name : str
            Name of the metric in ``daily_metrics``.
        contamination : float
            Expected proportion of anomalies (passed to IsolationForest).
        lookback_days : int
            Number of trailing days to analyze.

        Returns
        -------
        list[AnomalyResult]
            One entry per anomalous day, sorted by date.
        """
        raw = self._fetch_metric_series(metric_name, lookback_days)
        df = self._build_features(raw)

        feature_cols = [
            "metric_value",
            "day_of_week",
            "rolling_mean_7d",
            "rolling_std_7d",
            "pct_change",
            "z_score",
        ]
        X = df[feature_cols].values

        # --- Isolation Forest ---
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )
        df["iso_label"] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
        df["anomaly_score"] = iso.decision_function(X)

        # --- Control chart bounds (mean +/- 3 sigma from rolling window) ---
        overall_mean = df["metric_value"].mean()
        overall_std = df["metric_value"].std()
        ucl = overall_mean + 3 * overall_std
        lcl = overall_mean - 3 * overall_std
        df["is_cc_anomaly"] = (df["metric_value"] > ucl) | (df["metric_value"] < lcl)

        # Combine: flag if either method flags the point
        anomalies_mask = (df["iso_label"] == -1) | df["is_cc_anomaly"]
        anomaly_df = df[anomalies_mask]

        results: list[AnomalyResult] = []
        for _, row in anomaly_df.iterrows():
            results.append(
                AnomalyResult(
                    metric_date=row["metric_date"].strftime("%Y-%m-%d"),
                    metric_value=round(float(row["metric_value"]), 4),
                    expected_value=round(float(row["rolling_mean_7d"]), 4),
                    z_score=round(float(row["z_score"]), 4),
                    anomaly_score=round(float(row["anomaly_score"]), 4),
                    is_control_chart_anomaly=bool(row["is_cc_anomaly"]),
                    ucl=round(ucl, 4),
                    lcl=round(lcl, 4),
                )
            )

        logger.info(
            "Anomaly detection for '{}': {} anomalies in {} days "
            "(UCL={:.2f}, LCL={:.2f})",
            metric_name,
            len(results),
            len(df),
            ucl,
            lcl,
        )

        return results

    def detect_realtime_anomaly(
        self, metric_name: str, current_value: float
    ) -> bool:
        """Check whether a single new observation is anomalous.

        Compares ``current_value`` against the control-chart bounds
        (mean +/- 3 sigma) derived from the most recent 90 days of data.

        Parameters
        ----------
        metric_name : str
        current_value : float

        Returns
        -------
        bool
            True if the value falls outside control limits.
        """
        raw = self._fetch_metric_series(metric_name, lookback_days=90)
        mean = float(raw["metric_value"].mean())
        std = float(raw["metric_value"].std())
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        is_anomaly = current_value > ucl or current_value < lcl

        logger.info(
            "Realtime anomaly check for '{}': value={:.2f}, UCL={:.2f}, "
            "LCL={:.2f}, anomaly={}",
            metric_name,
            current_value,
            ucl,
            lcl,
            is_anomaly,
        )
        return is_anomaly
