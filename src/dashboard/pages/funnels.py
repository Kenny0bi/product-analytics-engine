"""Funnel page: the conversion river.

The standard funnel chart (stacked trapezoids) hides the thing that matters:
where people actually leave. Here the funnel is drawn as a river seen from
the side. The surviving flow is a teal ribbon that narrows at each gate; at
every gate the lost users peel off as a fading vermilion distributary,
labeled with exactly how many left and what share that was. The ribbon's
thickness is the metric, so the overall conversion rate is literally visible
as how much of the river reaches the end.

Below the river: funnel conversion compared across segments, and the median
time users take between steps.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analytics.funnels import FunnelAnalyzer
from src.dashboard import theme as T

_DEFAULT_STEPS = [
    "view_homepage", "view_product", "click_add_to_cart",
    "begin_checkout", "complete_purchase",
]

_SEGMENT_FIELDS = {
    "None": None,
    "Device type": "device_type",
    "UTM source": "utm_source",
    "Country": "country",
}


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Cubic smoothstep for organic ribbon transitions between gates."""
    return t * t * (3.0 - 2.0 * t)


def _ribbon_width(x: np.ndarray, widths: list[float]) -> np.ndarray:
    """Interpolate ribbon width at positions x given per-gate widths."""
    seg = np.clip(np.floor(x).astype(int), 0, len(widths) - 2)
    t = _smoothstep(x - seg)
    w0 = np.array([widths[i] for i in seg])
    w1 = np.array([widths[i + 1] for i in seg])
    return w0 * (1 - t) + w1 * t


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _river_chart(steps: list[str], users: list[int]) -> go.Figure:
    """Draw the conversion river for the given steps and user counts."""
    n = len(steps)
    w0 = max(users[0], 1)
    widths = [u / w0 for u in users]

    fig = T.base_figure(height=520)

    # --- Surviving ribbon ---
    xs = np.linspace(0, n - 1, 60 * (n - 1) + 1)
    wt = _ribbon_width(xs, widths)
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([np.zeros_like(xs), -wt[::-1]]),
        fill="toself", mode="lines",
        line=dict(color=T.TEAL, width=2),
        fillcolor=_rgba(T.TEAL, 0.55),
        hoverinfo="skip", showlegend=False,
    ))

    # --- Distributaries: the lost flow at each gate ---
    for i in range(n - 1):
        lost = widths[i] - widths[i + 1]
        if lost <= 0:
            continue
        # The wisp starts attached to the ribbon's underside at the gate and
        # drifts down-right while thinning away.
        xr = np.linspace(i, i + 0.62, 40)
        t = (xr - i) / 0.62
        surv_bottom = -_ribbon_width(xr, widths)
        drift = 1.35 * _smoothstep(t) * (0.20 + lost)   # separation grows from 0
        thick = lost * (1.0 - 0.72 * t)                 # taper away
        top = surv_bottom - drift
        bot = top - thick
        fig.add_trace(go.Scatter(
            x=np.concatenate([xr, xr[::-1]]),
            y=np.concatenate([top, bot[::-1]]),
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor=_rgba(T.VERMILION, 0.20),
            hoverinfo="skip", showlegend=False,
        ))
        lost_users = users[i] - users[i + 1]
        lost_pct = lost_users / users[i] if users[i] else 0
        fig.add_annotation(
            x=i + 0.66, y=float(bot[-1]) - 0.015,
            text=(f"&#8722;{T.fmt_compact(lost_users)} "
                  f"({lost_pct:.0%}) left here"),
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(family=T.FONT_MONO, size=11, color=T.CRIMSON),
        )

    # --- Gates, names, counts ---
    max_drop = max(0.25 + (widths[i] - widths[i + 1]) for i in range(n - 1)) \
        if n > 1 else 0.3
    floor = -1.0 - max_drop - 0.35
    for i, (name, u) in enumerate(zip(steps, users)):
        fig.add_shape(type="line", x0=i, x1=i, y0=0.02, y1=-widths[i] - 0.02,
                      line=dict(color=T.INK, width=1))
        fig.add_annotation(
            x=i, y=0.16, text=name.replace("_", " "),
            showarrow=False, font=dict(family=T.FONT_UI, size=12.5, color=T.INK),
            xanchor="center",
        )
        pct = u / users[0] if users[0] else 0
        fig.add_annotation(
            x=i, y=0.07,
            text=f"<b>{T.fmt_compact(u)}</b> &#183; {pct:.1%}",
            showarrow=False,
            font=dict(family=T.FONT_MONO, size=11.5, color=T.INK_MUTED),
            xanchor="center",
        )

    # --- Step-to-step continuation labels on the ribbon ---
    for i in range(n - 1):
        rate = users[i + 1] / users[i] if users[i] else 0
        mid_w = _ribbon_width(np.array([i + 0.5]), widths)[0]
        inside = mid_w > 0.18
        fig.add_annotation(
            x=i + 0.5, y=-mid_w / 2 if inside else -mid_w - 0.055,
            text=f"{rate:.0%} continue",
            showarrow=False,
            font=dict(family=T.FONT_MONO, size=11,
                      color=T.PAPER if inside else T.INK_MUTED),
        )

    fig.update_xaxes(visible=False, range=[-0.35, n - 1 + 1.05])
    fig.update_yaxes(visible=False, range=[floor, 0.30])
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    return fig


def _comparison_chart(results: dict, steps: list[str]) -> go.Figure:
    """Overall conversion by step, one line per segment, directly labeled."""
    fig = T.base_figure(height=360)
    end_labels = []
    for si, (seg_val, res) in enumerate(results.items()):
        color = T.CATEGORICAL[si % len(T.CATEGORICAL)]
        rates = [s.overall_conversion_rate for s in res.steps]
        fig.add_trace(go.Scatter(
            x=list(range(len(steps))), y=rates,
            mode="lines+markers", name=str(seg_val),
            line=dict(color=color, width=2.2),
            marker=dict(size=8, color=color,
                        line=dict(color=T.PAPER, width=2)),
            hovertemplate=(f"{seg_val}<br>%{{text}}: %{{y:.1%}} of entrants"
                           "<extra></extra>"),
            text=[s.replace("_", " ") for s in steps],
        ))
        end_labels.append((rates[-1], str(seg_val), color))

    # Direct end labels, pushed apart so small final values don't collide.
    min_gap = 0.05
    end_labels.sort(reverse=True)
    placed: list[float] = []
    for rate, seg_val, color in end_labels:
        y = rate
        if placed and placed[-1] - y < min_gap:
            y = placed[-1] - min_gap
        placed.append(y)
        fig.add_annotation(
            x=len(steps) - 1 + 0.06, y=max(y, 0.0),
            text=f"{seg_val} {rate:.1%}", showarrow=False, xanchor="left",
            font=dict(family=T.FONT_MONO, size=11, color=color),
        )
    fig.update_xaxes(
        tickvals=list(range(len(steps))),
        ticktext=[s.replace("_", " ") for s in steps],
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(showlegend=True, margin=dict(r=110))
    return fig


def render(conn) -> None:
    """Render the funnel page."""
    T.inject_css(st)
    if conn is None:
        st.error("No database found. Run `python -m src.main generate` first.")
        return

    T.headline(
        st, "Conversion", "Where the river narrows",
        "Every visitor enters on the left. Teal is the flow that continues; each "
        "vermilion distributary is the exact crowd that left at that gate, within "
        "a single session, steps taken in order.",
    )

    event_names = [r[0] for r in conn.execute(
        "select distinct event_name from events order by 1").fetchall()]

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        steps = st.multiselect("Funnel steps, in order", options=event_names,
                               default=[s for s in _DEFAULT_STEPS
                                        if s in event_names])
    with c2:
        min_d, max_d = conn.execute(
            "select min(cast(timestamp as date)), max(cast(timestamp as date)) "
            "from events").fetchone()
        drange = st.date_input("Date range", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)
    with c3:
        seg_label = st.selectbox("Compare by segment",
                                 list(_SEGMENT_FIELDS.keys()))

    if len(steps) < 2:
        st.warning("Pick at least two steps.")
        return

    start = str(drange[0]) if isinstance(drange, tuple) and drange else None
    end = str(drange[1]) if isinstance(drange, tuple) and len(drange) > 1 else None

    analyzer = FunnelAnalyzer(conn)
    result = analyzer.compute_funnel(steps=steps, start_date=start, end_date=end)
    users = [s.users for s in result.steps]

    T.kpi_tiles(st, [
        {"label": "Entered", "value": T.fmt_compact(result.total_entered)},
        {"label": "Converted", "value": T.fmt_compact(result.total_converted)},
        {"label": "End-to-end conversion",
         "value": f"{result.overall_conversion_rate:.2%}"},
        {"label": "Biggest leak", "value": _biggest_leak(result)},
    ])

    st.plotly_chart(_river_chart(steps, users), width='stretch')

    # Median time between steps.
    timings = [s.median_time_to_next for s in result.steps[:-1]]
    if any(t is not None for t in timings):
        st.markdown("**How long each crossing takes** &middot; median time "
                    "between consecutive steps, converters only")
        cols = st.columns(len(timings))
        for col, (i, t) in zip(cols, enumerate(timings)):
            label = (f"{steps[i].replace('_', ' ')} &#8594; "
                     f"{steps[i + 1].replace('_', ' ')}")
            val = T.fmt_duration(t) if t is not None else "n/a"
            col.markdown(
                f'<div class="pae-tile"><div class="k">{label}</div>'
                f'<div class="v">{val}</div></div>', unsafe_allow_html=True)

    # Segment comparison.
    field = _SEGMENT_FIELDS[seg_label]
    if field:
        T.rule(st)
        st.markdown(f"**The same river, split by {seg_label.lower()}** &middot; "
                    "share of entrants reaching each step")
        values = [r[0] for r in conn.execute(f"""
            select {field}, count(*) as n from events
            where {field} is not null
            group by 1 order by n desc limit 4
        """).fetchall()]
        comparison = analyzer.compare_funnels(steps, field, values)
        st.plotly_chart(_comparison_chart(comparison, steps),
                        width='stretch')


def _biggest_leak(result) -> str:
    """Name the step transition with the largest drop-off count."""
    worst = max(result.steps[1:], key=lambda s: s.dropoff_count, default=None)
    if worst is None:
        return "n/a"
    prev = result.steps[worst.step_index - 1]
    return (f"{prev.step_name.replace('_', ' ')} "
            f"({worst.dropoff_rate:.0%} lost)")
