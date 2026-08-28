"""Segments page: the population, person by person.

RFM tab: every user is a dot placed by when they last showed up (x, reversed
so "drifting away" reads left-to-right) and how often they come (y), sized
by what they've spent. Segment identity is the categorical hue, labeled
directly on the plot at each segment's center of mass. A share strip above
shows how the whole population divides.

Clusters tab: K-means fingerprints. Radar charts hide more than they show,
so each cluster's profile is a column of diverging bars: how many standard
deviations that cluster sits above or below the population on every
behavioral feature. A PCA scatter shows how cleanly the clusters separate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analytics.segmentation import SegmentationAnalyzer
from src.dashboard import theme as T

_SEGMENT_ORDER = ["Champions", "Loyal", "New", "At Risk", "Hibernating", "Other"]
_SEGMENT_COLORS = dict(zip(_SEGMENT_ORDER, T.CATEGORICAL))

_FEATURE_COLS = [
    "total_sessions", "total_events", "total_page_views",
    "avg_session_duration", "avg_events_per_session",
    "total_revenue", "num_purchases",
    "days_since_signup", "days_since_last_activity",
    "unique_features_used", "weekend_ratio",
]


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


@st.cache_data(ttl=3600, show_spinner="Scoring 50K users...")
def _rfm(_conn) -> pd.DataFrame:
    return SegmentationAnalyzer(_conn).compute_rfm()


@st.cache_data(ttl=3600, show_spinner="Clustering users...")
def _clusters(_conn, k: int) -> pd.DataFrame:
    return SegmentationAnalyzer(_conn).behavioral_clustering(n_clusters=k)


# ---------------------------------------------------------------------------
# RFM
# ---------------------------------------------------------------------------

def _share_strip(rfm: pd.DataFrame) -> go.Figure:
    """One horizontal strip: the population divided into segments."""
    counts = rfm["rfm_segment"].value_counts()
    total = counts.sum()
    fig = T.base_figure(height=110)
    x0 = 0.0
    for seg in _SEGMENT_ORDER:
        n = int(counts.get(seg, 0))
        if n == 0:
            continue
        share = n / total
        color = _SEGMENT_COLORS[seg]
        fig.add_trace(go.Bar(
            x=[share], y=[""], orientation="h", name=seg,
            marker=dict(color=color, line=dict(color=T.PAPER, width=2)),
            hovertemplate=(f"{seg}: {T.fmt_int(n)} users "
                           f"({share:.1%})<extra></extra>"),
        ))
        label = f"{seg} {share:.0%}" if share > 0.07 else f"{share:.0%}"
        fig.add_annotation(
            x=x0 + share / 2, y=0.0, text=label, showarrow=False,
            font=dict(family=T.FONT_MONO, size=11,
                      color=T.PAPER if share > 0.04 else T.INK_MUTED),
            yshift=0 if share > 0.04 else 26,
        )
        x0 += share
    fig.update_layout(barmode="stack", showlegend=False,
                      margin=dict(l=8, r=8, t=6, b=6))
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False)
    return fig


def _rfm_scatter(rfm: pd.DataFrame, sample_per_seg: int = 900) -> go.Figure:
    """Recency vs frequency, sized by monetary, one dot per (sampled) user."""
    fig = T.base_figure(height=520)
    rng = np.random.default_rng(42)
    max_monetary = max(float(rfm["monetary"].max()), 1.0)

    for seg in _SEGMENT_ORDER:
        seg_df = rfm[rfm["rfm_segment"] == seg]
        if seg_df.empty:
            continue
        if len(seg_df) > sample_per_seg:
            seg_df = seg_df.iloc[rng.choice(len(seg_df), sample_per_seg,
                                            replace=False)]
        color = _SEGMENT_COLORS[seg]
        sizes = 4 + 16 * np.sqrt(seg_df["monetary"].values / max_monetary)
        fig.add_trace(go.Scattergl(
            x=seg_df["recency_days"], y=seg_df["frequency"],
            mode="markers", name=seg,
            marker=dict(size=sizes, color=_rgba(color, 0.55),
                        line=dict(width=0)),
            customdata=seg_df["monetary"].values.round(0),
            hovertemplate=(f"{seg}<br>last seen %{{x}} days ago &#183; "
                           "%{y} sessions &#183; $%{customdata:,.0f} spent"
                           "<extra></extra>"),
        ))

    # Direct labels at each segment's center of mass.
    for seg in _SEGMENT_ORDER:
        seg_df = rfm[rfm["rfm_segment"] == seg]
        if len(seg_df) < 20:
            continue
        # On a log y-axis plotly expects annotation y as log10(value).
        fig.add_annotation(
            x=float(seg_df["recency_days"].median()),
            y=float(np.log10(max(seg_df["frequency"].median(), 1.0))),
            text=f"<b>{seg}</b>", showarrow=False,
            font=dict(family=T.FONT_DISPLAY, size=15,
                      color=_SEGMENT_COLORS[seg]),
            bgcolor=_rgba(T.PAPER, 0.75), borderpad=2,
        )

    fig.update_xaxes(title="days since last activity (drifting away &#8594;)")
    fig.update_yaxes(title="lifetime sessions", type="log",
                     tickvals=[1, 3, 10, 30, 100, 300],
                     ticktext=["1", "3", "10", "30", "100", "300"])
    fig.update_layout(legend=dict(itemsizing="constant"))
    return fig


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

def _fingerprint_chart(df: pd.DataFrame) -> go.Figure:
    """Small-multiple diverging bars: each cluster's z-score profile."""
    means = df[_FEATURE_COLS].mean()
    stds = df[_FEATURE_COLS].std().replace(0, 1)

    labels = (df.groupby("cluster_label")["user_id"].count()
              .sort_values(ascending=False).index.tolist())
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=len(labels),
        subplot_titles=[
            f"{lb}<br><span style='font-size:10px;color:{T.INK_MUTED}'>"
            f"{T.fmt_compact((df['cluster_label'] == lb).sum())} users</span>"
            for lb in labels],
        shared_yaxes=True, horizontal_spacing=0.015,
    )
    display_names = {
        "total_sessions": "sessions", "total_events": "events",
        "total_page_views": "page views",
        "avg_session_duration": "session length",
        "avg_events_per_session": "events / session",
        "total_revenue": "revenue", "num_purchases": "purchases",
        "days_since_signup": "tenure",
        "days_since_last_activity": "days inactive",
        "unique_features_used": "features used",
        "weekend_ratio": "weekend share",
    }
    feature_names = [display_names[c] for c in _FEATURE_COLS]

    for ci, lb in enumerate(labels):
        sub = df[df["cluster_label"] == lb]
        z = ((sub[_FEATURE_COLS].mean() - means) / stds).clip(-3, 3)
        colors = [T.TEAL if v >= 0 else T.VERMILION for v in z.values]
        fig.add_trace(go.Bar(
            x=z.values, y=feature_names, orientation="h",
            marker=dict(color=colors, line=dict(color=T.PAPER, width=1)),
            hovertemplate=(f"{lb}<br>%{{y}}: %{{x:.2f}} sd from population"
                           "<extra></extra>"),
            showlegend=False,
        ), row=1, col=ci + 1)
        fig.add_vline(x=0, line=dict(color=T.LINE_SOFT, width=1),
                      row=1, col=ci + 1)

    fig.update_layout(template="pae", height=420,
                      margin=dict(l=120, r=8, t=64, b=30))
    fig.update_xaxes(range=[-3.2, 3.2], tickvals=[-2, 0, 2],
                     tickfont=dict(family=T.FONT_MONO, size=10))
    fig.update_yaxes(tickfont=dict(size=10.5), autorange="reversed")
    for ann in fig.layout.annotations:
        ann.font = dict(family=T.FONT_DISPLAY, size=13, color=T.INK)
    return fig


def _pca_chart(df: pd.DataFrame, sample_n: int = 4000) -> go.Figure:
    """2-D PCA projection of the behavioral feature space, by cluster."""
    X = StandardScaler().fit_transform(df[_FEATURE_COLS].fillna(0).values)
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X)
    plot_df = pd.DataFrame({
        "x": proj[:, 0], "y": proj[:, 1], "label": df["cluster_label"].values,
    })
    rng = np.random.default_rng(42)
    if len(plot_df) > sample_n:
        plot_df = plot_df.iloc[rng.choice(len(plot_df), sample_n,
                                          replace=False)]

    fig = T.base_figure(height=440)
    labels = (df.groupby("cluster_label")["user_id"].count()
              .sort_values(ascending=False).index.tolist())
    for ci, lb in enumerate(labels):
        sub = plot_df[plot_df["label"] == lb]
        color = T.CATEGORICAL[ci % len(T.CATEGORICAL)]
        fig.add_trace(go.Scattergl(
            x=sub["x"], y=sub["y"], mode="markers", name=lb,
            marker=dict(size=5, color=_rgba(color, 0.5), line=dict(width=0)),
            hovertemplate=f"{lb}<extra></extra>",
        ))
    var = pca.explained_variance_ratio_
    fig.update_xaxes(title=f"PC1 ({var[0]:.0%} of variance)")
    fig.update_yaxes(title=f"PC2 ({var[1]:.0%} of variance)")
    fig.update_layout(legend=dict(itemsizing="constant"))
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render(conn) -> None:
    """Render the segmentation page."""
    T.inject_css(st)
    if conn is None:
        st.error("No database found. Run `python -m src.main generate` first.")
        return

    T.headline(
        st, "Segmentation", "The population, person by person",
        "Two independent reads on the same 50K users: RFM quintile scoring "
        "(who is valuable right now) and K-means behavioral clustering "
        "(who acts alike, regardless of value).",
    )

    tab_rfm, tab_clusters = st.tabs(["RFM segments", "Behavioral clusters"])

    with tab_rfm:
        rfm = _rfm(conn)
        st.markdown("**How the population divides**")
        st.plotly_chart(_share_strip(rfm), width='stretch')

        st.markdown(
            "**Every user, placed** &middot; a stratified sample of up to 900 "
            "dots per segment; dot area is lifetime spend. Champions live "
            "bottom-left-up (recent and frequent); Hibernating drifts right."
        )
        st.plotly_chart(_rfm_scatter(rfm), width='stretch')

        profiles = SegmentationAnalyzer(conn).compute_rfm_profiles(rfm)
        st.markdown("**Segment profiles**")
        st.dataframe(pd.DataFrame([{
            "Segment": p.segment,
            "Users": T.fmt_int(p.user_count),
            "Share": f"{p.pct_of_total:.1f}%",
            "Median days since seen": f"{p.avg_recency_days:.0f}",
            "Avg sessions": f"{p.avg_frequency:.1f}",
            "Avg lifetime spend": f"${p.avg_monetary:,.0f}",
        } for p in profiles]), width='stretch', hide_index=True)

    with tab_clusters:
        k = st.slider("Number of clusters (k)", 3, 8, 5)
        clustered = _clusters(conn, k)

        st.markdown(
            "**Cluster fingerprints** &middot; each column is one cluster; "
            "bars show how far it sits from the population mean on every "
            "behavioral feature, in standard deviations. Teal = above, "
            "vermilion = below."
        )
        st.plotly_chart(_fingerprint_chart(clustered),
                        width='stretch')

        T.rule(st)
        st.markdown(
            "**Do the clusters actually separate?** &middot; 2-D PCA "
            "projection of the 11-feature space"
        )
        st.plotly_chart(_pca_chart(clustered), width='stretch')
