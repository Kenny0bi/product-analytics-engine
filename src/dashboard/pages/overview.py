"""Overview page: the current of users moving through the product.

Three pieces:

1. KPI tiles: DAU, WAU, MAU, 30-day revenue, conversion, session length,
   bounce rate, each with a week-over-week delta where it means something.
2. "The current": daily active users over the trailing 90 days, drawn as a
   filled stream with weekend shading, a 7-day rolling centerline, and the
   peak day annotated. Revenue gets the same treatment beside it.
3. "The week's pulse": event volume by hour of day, one ridge per weekday.
   The product's daily rhythm (10 AM and 8 PM peaks, quiet weekends) is
   the actual finding, so the chart is built to show exactly that shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import theme as T

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _kpis(conn) -> dict:
    """Headline KPIs plus week-over-week comparators, in one SQL pass each."""
    # Anchor on the last complete day: the newest date is structurally
    # partial and would report a misleadingly tiny DAU.
    ref = conn.execute(
        "select max(cast(timestamp as date)) - 1 from events").fetchone()[0]
    row = conn.execute("""
        with ref as (select max(cast(timestamp as date)) - 1 as d from events)
        select
            (select count(distinct user_id) from events, ref
             where cast(timestamp as date) = ref.d) as dau,
            (select count(distinct user_id) from events, ref
             where cast(timestamp as date) > ref.d - 7) as wau,
            (select count(distinct user_id) from events, ref
             where cast(timestamp as date) > ref.d - 30) as mau,
            (select count(distinct user_id) from events, ref
             where cast(timestamp as date) > ref.d - 14
               and cast(timestamp as date) <= ref.d - 7) as wau_prev,
            (select coalesce(sum(revenue), 0) from events, ref
             where cast(timestamp as date) > ref.d - 30) as revenue_30d,
            (select coalesce(sum(revenue), 0) from events, ref
             where cast(timestamp as date) > ref.d - 60
               and cast(timestamp as date) <= ref.d - 30) as revenue_prev,
            (select avg(case when has_conversion then 1.0 else 0.0 end)
             from sessions, ref where cast(started_at as date) > ref.d - 30) as conv,
            (select avg(duration_seconds) from sessions, ref
             where cast(started_at as date) > ref.d - 30) as avg_dur,
            (select avg(case when event_count = 1 and page_view_count = 1
                        then 1.0 else 0.0 end)
             from sessions, ref where cast(started_at as date) > ref.d - 30) as bounce
    """).fetchone()
    return {
        "ref_date": ref, "dau": row[0], "wau": row[1], "mau": row[2],
        "wau_prev": row[3], "revenue_30d": row[4], "revenue_prev": row[5],
        "conv": row[6] or 0, "avg_dur": row[7] or 0, "bounce": row[8] or 0,
    }


def _daily_series(conn, days: int = 90) -> pd.DataFrame:
    """Daily active users and revenue for the trailing window."""
    return conn.execute(f"""
        with ref as (select max(cast(timestamp as date)) - 1 as d from events)
        select
            cast(e.timestamp as date) as day,
            count(distinct e.user_id) as dau,
            coalesce(sum(e.revenue), 0) as revenue
        from events e, ref
        where cast(e.timestamp as date) > ref.d - {days}
          and cast(e.timestamp as date) <= ref.d
        group by 1
        order by 1
    """).fetchdf()


def _weekly_pulse(conn) -> pd.DataFrame:
    """Event volume by (weekday, hour) over the trailing 90 days."""
    return conn.execute("""
        with ref as (select max(cast(timestamp as date)) as d from events)
        select
            dayofweek(e.timestamp) as dow,   -- 0 = Sunday in DuckDB
            hour(e.timestamp) as hr,
            count(*) as events
        from events e, ref
        where cast(e.timestamp as date) > ref.d - 90
        group by 1, 2
        order by 1, 2
    """).fetchdf()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _stream_chart(df: pd.DataFrame, ycol: str, color: str, label: str,
                  money: bool = False) -> go.Figure:
    """A daily series as a soft stream: fill, rolling centerline, peak flag."""
    fig = T.base_figure(height=320)
    days = pd.to_datetime(df["day"])
    y = df[ycol].astype(float)
    roll = y.rolling(7, min_periods=1).mean()

    # Weekend shading: one faint band per Saturday-Sunday pair.
    for d in days:
        if d.dayofweek == 5:
            fig.add_vrect(x0=d, x1=d + pd.Timedelta(days=2),
                          fillcolor=T.LINE_FAINT, opacity=0.45, line_width=0)

    hover = "%{x|%a %b %d}<br>" + label + ": %{y:$,.0f}<extra></extra>" if money \
        else "%{x|%a %b %d}<br>" + label + ": %{y:,.0f}<extra></extra>"

    fig.add_trace(go.Scatter(
        x=days, y=y, mode="lines", name=label,
        line=dict(color=color, width=1), opacity=0.55,
        fill="tozeroy",
        fillcolor=f"rgba{(*_hex_rgb(color), 0.12)}",
        hovertemplate=hover,
    ))
    fig.add_trace(go.Scatter(
        x=days, y=roll, mode="lines", name="7-day average",
        line=dict(color=color, width=2.6), hoverinfo="skip",
    ))

    # Peak-day flag.
    peak_i = int(y.idxmax())
    peak_val = "$" + T.fmt_compact(y[peak_i]) if money else T.fmt_compact(y[peak_i])
    fig.add_annotation(
        x=days[peak_i], y=y[peak_i], text=f"peak {peak_val}",
        showarrow=True, arrowhead=0, arrowcolor=T.INK_MUTED, ax=0, ay=-26,
        font=dict(family=T.FONT_MONO, size=11, color=T.INK),
    )
    fig.update_layout(showlegend=False, hovermode="x unified")
    fig.update_yaxes(rangemode="tozero",
                     tickprefix="$" if money else None)
    return fig


def _pulse_chart(pulse: pd.DataFrame) -> go.Figure:
    """Ridgeline of hourly event volume, one ridge per weekday, Mon at top.

    Each ridge is normalized to the busiest hour of the whole week so the
    weekday-versus-weekend volume difference stays honest.
    """
    fig = T.base_figure(height=420)
    # DuckDB dayofweek: 0=Sunday..6=Saturday. Reorder to Mon..Sun.
    dow_order = [1, 2, 3, 4, 5, 6, 0]
    grid = pulse.pivot_table(index="dow", columns="hr", values="events",
                             aggfunc="sum").reindex(dow_order).fillna(0)
    grid = grid.reindex(columns=range(24), fill_value=0)
    peak = grid.values.max() or 1
    gap = 1.0  # vertical distance between baselines
    amp = 1.55  # max ridge height in baseline units

    hours = np.arange(24)
    for row_i, dow in enumerate(dow_order):
        base = (len(dow_order) - 1 - row_i) * gap
        vals = grid.loc[dow].values / peak
        y = base + vals * amp
        weekend = dow in (0, 6)
        color = T.INK_MUTED if weekend else T.VERMILION
        fig.add_trace(go.Scatter(
            x=hours, y=[base] * 24, mode="lines",
            line=dict(color=T.LINE_SOFT, width=1), hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=hours, y=y, mode="lines", name=_WEEKDAYS[row_i],
            line=dict(color=color, width=2, shape="spline", smoothing=0.8),
            fill="tonexty",
            fillcolor=f"rgba{(*_hex_rgb(color), 0.14)}",
            customdata=np.stack([grid.loc[dow].values], axis=-1),
            hovertemplate=(_WEEKDAYS[row_i]
                           + " %{x}:00<br>%{customdata[0]:,.0f} events<extra></extra>"),
            showlegend=False,
        ))
        fig.add_annotation(
            x=-0.6, y=base + 0.12, text=_WEEKDAYS[row_i], showarrow=False,
            xanchor="right",
            font=dict(family=T.FONT_MONO, size=11,
                      color=T.INK_MUTED if weekend else T.INK),
        )

    # Mark the two structural peaks.
    for hr in (10, 20):
        fig.add_vline(x=hr, line=dict(color=T.LINE_SOFT, width=1, dash="dot"))
    fig.add_annotation(x=10, y=len(dow_order) * gap + 0.3, text="10 AM peak",
                       showarrow=False,
                       font=dict(family=T.FONT_MONO, size=10, color=T.INK_MUTED))
    fig.add_annotation(x=20, y=len(dow_order) * gap + 0.3, text="8 PM peak",
                       showarrow=False,
                       font=dict(family=T.FONT_MONO, size=10, color=T.INK_MUTED))

    fig.update_xaxes(tickvals=[0, 4, 8, 12, 16, 20, 23],
                     ticktext=["12a", "4a", "8a", "12p", "4p", "8p", "11p"],
                     showgrid=False, range=[-2.2, 23.5])
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=8, r=16, t=34, b=36))
    return fig


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render(conn) -> None:
    """Render the overview page."""
    T.inject_css(st)
    if conn is None:
        st.error(
            "No database found. Run `python -m src.main generate` first: this "
            "dashboard only ever shows real query results, never canned demo numbers."
        )
        return

    k = _kpis(conn)
    T.headline(
        st, "Product Analytics Engine", "The state of the current",
        f"50K synthetic users, one year of product telemetry. Everything below is "
        f"computed live from DuckDB as of {k['ref_date']}.",
    )

    wau_delta = (k["wau"] - k["wau_prev"]) / k["wau_prev"] if k["wau_prev"] else 0
    rev_delta = ((k["revenue_30d"] - k["revenue_prev"]) / k["revenue_prev"]
                 if k["revenue_prev"] else 0)
    T.kpi_tiles(st, [
        {"label": "DAU", "value": T.fmt_compact(k["dau"])},
        {"label": "WAU", "value": T.fmt_compact(k["wau"]),
         "delta": f"{wau_delta:+.1%} WoW",
         "direction": "up" if wau_delta >= 0 else "down"},
        {"label": "MAU", "value": T.fmt_compact(k["mau"])},
        {"label": "Revenue 30d", "value": T.fmt_money(k["revenue_30d"]),
         "delta": f"{rev_delta:+.1%} vs prior 30d",
         "direction": "up" if rev_delta >= 0 else "down"},
        {"label": "Session conversion", "value": f"{k['conv']:.1%}"},
        {"label": "Avg session", "value": T.fmt_duration(k["avg_dur"])},
        {"label": "Bounce rate", "value": f"{k['bounce']:.1%}"},
    ])

    daily = _daily_series(conn, days=90)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daily active users** · last 90 days, weekends shaded")
        st.plotly_chart(_stream_chart(daily, "dau", T.BLUE, "Active users"),
                        width='stretch')
    with col2:
        st.markdown("**Daily revenue** · last 90 days")
        st.plotly_chart(_stream_chart(daily, "revenue", T.TEAL, "Revenue",
                                      money=True),
                        width='stretch')

    T.rule(st)
    st.markdown(
        "**The week's pulse** · event volume by hour, last 90 days. "
        "Weekdays in vermilion, weekends in gray; every ridge shares one scale, "
        "so flatter weekend ridges are genuinely quieter days."
    )
    st.plotly_chart(_pulse_chart(_weekly_pulse(conn)), width='stretch')
