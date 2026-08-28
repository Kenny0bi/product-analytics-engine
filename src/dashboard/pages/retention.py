"""Retention page: cohorts as comet trails.

The usual retention view is a red-to-green heatmap, which mostly measures
how bright your monitor is. Here each cohort is a comet: it enters at
period 0 at full size and its trail fades exactly as fast as its users
stop coming back. Mark area and opacity both encode the retention rate,
so a cohort that holds its users stays visible far to the right, and a
leaky one burns out early. Reading down a column compares cohorts at the
same age; reading along a row is one cohort's whole life.

Below the trails: the average retention curve with its interquartile band
across cohorts, which is the product's characteristic decay shape.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analytics.retention import RetentionAnalyzer
from src.dashboard import theme as T


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _comet_chart(matrix, annotate: bool) -> go.Figure:
    """Cohort comet trails: size + opacity encode the retention rate."""
    cohorts = matrix.cohorts
    periods = matrix.periods
    n_rows = len(cohorts)
    fig = T.base_figure(height=max(360, 34 * n_rows + 110))

    for row_i, cohort in enumerate(cohorts):
        y = n_rows - 1 - row_i
        rates = matrix.retention_rates.get(cohort, [])
        size = matrix.cohort_sizes.get(cohort, 0)

        # Faint guide line for the row.
        fig.add_trace(go.Scatter(
            x=[0, len(periods) - 1], y=[y, y], mode="lines",
            line=dict(color=T.LINE_FAINT, width=1),
            hoverinfo="skip", showlegend=False,
        ))

        xs, sizes, colors, texts = [], [], [], []
        for p, rate in zip(periods, rates):
            if rate <= 0:
                continue
            xs.append(p)
            # Area encodes the rate: diameter ~ sqrt(rate).
            sizes.append(6 + 20 * np.sqrt(rate / 100.0))
            colors.append(_rgba(T.TEAL, 0.25 + 0.75 * rate / 100.0))
            texts.append(f"{cohort} &#183; period {p}<br>{rate:.1f}% of "
                         f"{T.fmt_int(size)} users active")
        fig.add_trace(go.Scatter(
            x=xs, y=[y] * len(xs), mode="markers",
            marker=dict(size=sizes, color=colors,
                        line=dict(color=T.PAPER, width=1.5)),
            hovertemplate="%{text}<extra></extra>", text=texts,
            showlegend=False,
        ))

        # Row label: cohort and its size.
        fig.add_annotation(
            x=-0.55, y=y, xanchor="right",
            text=f"{cohort}  <span style='color:{T.INK_FAINT}'>"
                 f"{T.fmt_compact(size)}</span>",
            showarrow=False,
            font=dict(family=T.FONT_MONO, size=11, color=T.INK),
        )
        # Annotate the tail value (last nonzero period) for each cohort.
        if annotate and len(xs) > 1:
            last_p = xs[-1]
            last_rate = [r for r in rates if r > 0][-1]
            fig.add_annotation(
                x=last_p + 0.35, y=y, xanchor="left",
                text=f"{last_rate:.0f}%", showarrow=False,
                font=dict(family=T.FONT_MONO, size=10, color=T.INK_MUTED),
            )

    fig.update_xaxes(
        tickvals=periods, title=None, showgrid=False, zeroline=False,
        tickfont=dict(family=T.FONT_MONO, size=11),
        range=[-2.2, len(periods) + 0.6],
    )
    fig.update_yaxes(visible=False, range=[-0.7, n_rows - 0.3])
    fig.add_annotation(
        x=0, y=n_rows - 0.35, xanchor="center", yanchor="bottom",
        text="periods since signup &#8594;", showarrow=False,
        font=dict(family=T.FONT_UI, size=11, color=T.INK_FAINT),
    )
    fig.update_layout(margin=dict(l=10, r=30, t=26, b=36))
    return fig


def _curve_chart(analyzer: RetentionAnalyzer, matrix, period: str) -> go.Figure:
    """Average retention curve with the interquartile band across cohorts."""
    # Build per-period distributions from the matrix itself so the band uses
    # exactly the cohorts shown above.
    periods = matrix.periods
    per_period: list[list[float]] = [[] for _ in periods]
    for cohort in matrix.cohorts:
        for p, rate in zip(periods, matrix.retention_rates.get(cohort, [])):
            # A zero can mean "cohort too young to have this period"; drop
            # trailing structural zeros by requiring the cohort to be old
            # enough (rate>0 at any later period keeps ambiguity low for
            # synthetic data, so keep it simple: skip zeros beyond period 0).
            if p == 0 or rate > 0:
                per_period[p].append(rate)

    xs = [p for p in periods if per_period[p]]
    med = [float(np.median(per_period[p])) for p in xs]
    q1 = [float(np.percentile(per_period[p], 25)) for p in xs]
    q3 = [float(np.percentile(per_period[p], 75)) for p in xs]

    fig = T.base_figure(height=330)
    fig.add_trace(go.Scatter(
        x=xs + xs[::-1], y=q3 + q1[::-1], fill="toself", mode="lines",
        line=dict(width=0), fillcolor=_rgba(T.TEAL, 0.16),
        hoverinfo="skip", showlegend=False, name="IQR",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=med, mode="lines+markers", name="median cohort",
        line=dict(color=T.INK, width=2.4),
        marker=dict(size=7, color=T.INK, line=dict(color=T.PAPER, width=2)),
        hovertemplate=("period %{x}<br>median %{y:.1f}% "
                       "(band = cohort IQR)<extra></extra>"),
    ))
    for p in (1, min(3, len(xs) - 1)):
        if 0 < p < len(xs):
            fig.add_annotation(
                x=xs[p], y=med[p], text=f"{med[p]:.0f}%",
                showarrow=True, arrowhead=0, ax=0, ay=-24,
                arrowcolor=T.INK_MUTED,
                font=dict(family=T.FONT_MONO, size=11, color=T.INK),
            )
    fig.update_xaxes(title=f"{period}s since signup", tickvals=xs)
    fig.update_yaxes(title="active users, % of cohort", rangemode="tozero",
                     ticksuffix="%")
    fig.update_layout(showlegend=False)
    return fig


def render(conn) -> None:
    """Render the retention page."""
    T.inject_css(st)
    if conn is None:
        st.error("No database found. Run `python -m src.main generate` first.")
        return

    T.headline(
        st, "Retention", "How long cohorts keep burning",
        "One comet per signup cohort. Every trail starts at 100% and fades as "
        "its users stop showing up: both the dot's area and its ink encode the "
        "share still active, so a durable cohort is simply one you can still "
        "see far to the right.",
    )

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        period = st.radio("Cohort period", ["month", "week"], horizontal=True)
    with c2:
        num_periods = st.slider("Periods to show", 4, 24,
                                12 if period == "month" else 16)
    with c3:
        annotate = st.checkbox("Label trail ends", value=True)

    analyzer = RetentionAnalyzer(conn)
    matrix = analyzer.compute_retention(period=period, num_periods=num_periods)

    if period == "week" and len(matrix.cohorts) > 20:
        # Weekly cohorts over a year: keep the newest 20 trails readable.
        keep = matrix.cohorts[-20:]
        matrix.cohorts = keep
        matrix.retention_rates = {c: matrix.retention_rates[c] for c in keep}
        st.caption("Showing the 20 most recent weekly cohorts.")

    st.plotly_chart(_comet_chart(matrix, annotate), width='stretch')

    T.rule(st)
    st.markdown("**The decay curve** &middot; median retention across all "
                "cohorts, with the interquartile band")
    st.plotly_chart(_curve_chart(analyzer, matrix, period),
                    width='stretch')
