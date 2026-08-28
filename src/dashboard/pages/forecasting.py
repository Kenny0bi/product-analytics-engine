"""Forecasting page: the cone of futures.

History is a single ink line; the future is a cone that widens exactly as
fast as the model's certainty decays, drawn as nested 80% and 95% bands.
The seam between them is marked so nobody mistakes the fit for the fact.

Below the cone: anomaly flares. Days that Isolation Forest or the 3-sigma
control chart flags are drawn as haloed markers on the history, with the
control rails as faint guides, and each flare is listed with its severity.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import theme as T
from src.statistics.anomaly_detection import AnomalyDetector
from src.statistics.forecasting import MetricForecaster

_METRICS = {
    "Daily active users": "dau",
    "Revenue": "revenue",
    "Signups": "signups",
    "Conversion rate": "conversion_rate",
    "Avg session duration": "avg_session_duration",
    "Bounce rate": "bounce_rate",
}

_MODEL_NAMES = {
    "holt-winters": "Holt-Winters (additive trend + weekly seasonality)",
    "holt": "Holt linear trend",
    "naive": "naive last-value",
}


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _history(conn, metric: str, days: int = 180) -> pd.DataFrame:
    # The newest day is structurally incomplete; drop it so the history
    # line doesn't plunge to near zero right at the forecast seam.
    return conn.execute("""
        select metric_date, metric_value
        from daily_metrics
        where metric_name = ? and dimension_name = 'overall'
          and metric_date < (
              select max(metric_date) from daily_metrics
              where metric_name = ? and dimension_name = 'overall'
          )
        order by metric_date desc
        limit ?
    """, [metric, metric, days]).fetchdf().sort_values("metric_date")


def _cone_chart(hist: pd.DataFrame, fc, money: bool, pct: bool) -> go.Figure:
    """History line plus the widening forecast cone."""
    fig = T.base_figure(height=430)
    hx = pd.to_datetime(hist["metric_date"])
    hy = hist["metric_value"].astype(float)
    fx = pd.to_datetime(fc.forecast_dates)

    val_fmt = "$%{y:,.0f}" if money else ("%{y:.1%}" if pct else "%{y:,.0f}")

    # Counts, revenue, and rates cannot go below zero (rates not above one);
    # clip the interval bands to the metric's feasible range.
    hi_cap = 1.0 if pct else float("inf")

    def _clip(vals):
        return [min(max(v, 0.0), hi_cap) for v in vals]

    # 95% band (outer), then 80% (inner), then the center path.
    fig.add_trace(go.Scatter(
        x=list(fx) + list(fx[::-1]),
        y=_clip(fc.upper_95) + _clip(fc.lower_95)[::-1],
        fill="toself", mode="lines", line=dict(width=0),
        fillcolor=_rgba(T.VERMILION, 0.10), name="95% interval",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(fx) + list(fx[::-1]),
        y=_clip(fc.upper_80) + _clip(fc.lower_80)[::-1],
        fill="toself", mode="lines", line=dict(width=0),
        fillcolor=_rgba(T.VERMILION, 0.16), name="80% interval",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="lines", name="observed",
        line=dict(color=T.INK, width=1.8),
        hovertemplate="%{x|%b %d}: " + val_fmt + "<extra>observed</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fx, y=fc.forecast_values, mode="lines", name="forecast",
        line=dict(color=T.VERMILION, width=2.2),
        hovertemplate="%{x|%b %d}: " + val_fmt + "<extra>forecast</extra>",
    ))

    # The seam between fact and fit.
    seam = hx.iloc[-1]
    fig.add_vline(x=seam, line=dict(color=T.INK_MUTED, width=1, dash="dot"))
    fig.add_annotation(x=seam, y=1.06, yref="paper", text="forecast begins",
                       showarrow=False,
                       font=dict(family=T.FONT_MONO, size=10,
                                 color=T.INK_MUTED))

    fig.update_yaxes(tickprefix="$" if money else None,
                     tickformat=".0%" if pct else None, rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return fig


def _flare_chart(hist: pd.DataFrame, anomalies, money: bool,
                 pct: bool) -> go.Figure:
    """History with anomaly flares and control-chart rails."""
    fig = T.base_figure(height=380)
    hx = pd.to_datetime(hist["metric_date"])
    hy = hist["metric_value"].astype(float)
    roll = hy.rolling(7, min_periods=1).mean()
    val_fmt = "$%{y:,.0f}" if money else ("%{y:.1%}" if pct else "%{y:,.0f}")

    if anomalies:
        ucl, lcl = anomalies[0].ucl, anomalies[0].lcl
        for lvl, name in ((ucl, "UCL (+3 sigma)"), (lcl, "LCL (-3 sigma)")):
            fig.add_hline(y=lvl,
                          line=dict(color=T.LINE_SOFT, width=1.2, dash="dash"))
            fig.add_annotation(x=hx.iloc[2], y=lvl, text=name, showarrow=False,
                               yshift=9, xanchor="left",
                               font=dict(family=T.FONT_MONO, size=9.5,
                                         color=T.INK_FAINT))

    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="lines", name="observed",
        line=dict(color=T.INK, width=1.4), opacity=0.75,
        hovertemplate="%{x|%b %d}: " + val_fmt + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hx, y=roll, mode="lines", name="7-day mean",
        line=dict(color=T.INK_MUTED, width=1.4, dash="dot"),
        hoverinfo="skip",
    ))

    if anomalies:
        ax = [pd.Timestamp(a.metric_date) for a in anomalies]
        ay = [a.metric_value for a in anomalies]
        # Halo ring under the flare marker.
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="markers", name="anomaly",
            marker=dict(size=17, color=_rgba(T.CRIMSON, 0.18),
                        line=dict(color=T.CRIMSON, width=1)),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="markers", name="anomaly",
            marker=dict(size=7, color=T.CRIMSON,
                        line=dict(color=T.PAPER, width=1.5)),
            customdata=[[a.expected_value, a.z_score] for a in anomalies],
            hovertemplate=("%{x|%b %d}: " + val_fmt +
                           "<br>expected ~%{customdata[0]:,.1f} "
                           "(z = %{customdata[1]:.1f})<extra>anomaly</extra>"),
        ))

    fig.update_yaxes(tickprefix="$" if money else None,
                     tickformat=".0%" if pct else None)
    return fig


def render(conn) -> None:
    """Render the forecasting and anomaly page."""
    T.inject_css(st)
    if conn is None:
        st.error("No database found. Run `python -m src.main generate` first.")
        return

    T.headline(
        st, "Forecasting", "The cone of futures",
        "Holt-Winters exponential smoothing projected forward. The cone "
        "widens with the square root of the horizon: that widening is the "
        "honest part of the forecast.",
    )

    c1, c2 = st.columns([2, 2])
    with c1:
        metric_label = st.selectbox("Metric", list(_METRICS.keys()))
    with c2:
        horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
    metric = _METRICS[metric_label]
    money = metric == "revenue"
    pct = metric in ("conversion_rate", "bounce_rate")

    try:
        fc = MetricForecaster(conn).forecast_metric(
            metric_name=metric, periods_ahead=horizon)
    except ValueError as exc:
        st.warning(f"{exc}. Run `python -m src.main analyze` to materialize "
                   "daily metrics first.")
        return

    hist = _history(conn, metric)
    st.caption(f"Model: {_MODEL_NAMES.get(fc.model_type, fc.model_type)} "
               f"&middot; residual sigma = {fc.residual_std:,.2f}")
    st.plotly_chart(_cone_chart(hist, fc, money, pct),
                    width='stretch')

    T.rule(st)
    st.markdown(
        "**Anomaly flares** &middot; days flagged by Isolation Forest or the "
        "3-sigma control chart over the last 90 days"
    )
    anomalies = AnomalyDetector(conn).detect_metric_anomalies(
        metric_name=metric, lookback_days=90)
    st.plotly_chart(
        _flare_chart(_history(conn, metric, days=90), anomalies, money, pct),
        width='stretch')

    if anomalies:
        st.dataframe(pd.DataFrame([{
            "Date": a.metric_date,
            "Value": f"${a.metric_value:,.0f}" if money
                     else (f"{a.metric_value:.1%}" if pct
                           else f"{a.metric_value:,.1f}"),
            "Expected (7d mean)": f"{a.expected_value:,.1f}",
            "z-score": f"{a.z_score:+.1f}",
            "Beyond control limits": "yes" if a.is_control_chart_anomaly
                                     else "no",
            "Severity": ("high" if a.is_control_chart_anomaly
                         or abs(a.z_score) >= 4
                         else "medium" if abs(a.z_score) >= 2.5 else "low"),
        } for a in anomalies]), width='stretch', hide_index=True)
    else:
        st.caption("No anomalies flagged in the window. The metric is "
                   "behaving.")
