"""Design system for the Product Analytics Engine dashboard.

The dashboard reads as an editorial piece about the current of users moving
through a product, not as a stock BI tool. That intent drives every choice
here:

- Warm paper surface with warm near-black ink, instead of the default
  dark-dashboard look every analytics tool ships with.
- A six-hue categorical palette validated for colorblind separation,
  lightness band, chroma, and contrast against the paper surface
  (all six checks pass; see the palette constants below).
- Fraunces for display headings, Inter for UI text, IBM Plex Mono for
  figures, so numbers read like instrument readouts.
- Chart forms built from the shape of the data itself: ribbon funnels
  with named distributaries, cohort comet trails, posterior landscapes,
  SPRT decision corridors, forecast fans.

Every page imports from this module; nothing defines its own colors.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Surfaces and ink
# ---------------------------------------------------------------------------
PAPER = "#FAF6EF"          # warm paper background
PANEL = "#F3EDE1"          # slightly deeper panel surface
INK = "#201A14"            # warm near-black text
INK_MUTED = "#6B6156"      # secondary text
INK_FAINT = "#A89C8C"      # tertiary text / disabled
LINE_FAINT = "#E4DCCE"     # hairline grid and rules
LINE_SOFT = "#D5CBB9"      # slightly stronger rules

# ---------------------------------------------------------------------------
# Categorical palette (fixed order, never cycled).
# Validated on PAPER: lightness band, chroma floor, CVD separation
# (worst adjacent pair deltaE 11.8 deutan), normal-vision floor
# (worst 23.7), contrast >= 3:1. All six checks pass.
# ---------------------------------------------------------------------------
VERMILION = "#E4572E"
BLUE = "#1F7AC0"
AMBER = "#C77D0A"
PLUM = "#7B4B94"
TEAL = "#1B998B"
CRIMSON = "#A23E48"

CATEGORICAL: list[str] = [VERMILION, BLUE, AMBER, PLUM, TEAL, CRIMSON]

# ---------------------------------------------------------------------------
# Sequential ramp (magnitude): single teal hue, light -> dark.
# ---------------------------------------------------------------------------
SEQUENTIAL: list[str] = [
    "#E7F3F0", "#C4E4DD", "#9BD1C6", "#6FBAAB",
    "#43A08E", "#1B998B", "#147A6E", "#0E5C53", "#093F39",
]

# Diverging pair (polarity): vermilion <-> teal through a warm neutral.
DIVERGING: list[str] = [
    "#B23A16", "#E4572E", "#F0906E", "#F5C4B2",
    "#CFC8BB",
    "#A9D8CF", "#5FB8A8", "#1B998B", "#0E5C53",
]

# Status (reserved; never used as "series 4"). Ship with icon + label.
GOOD = "#147A6E"
WARNING = "#C77D0A"
SERIOUS = "#A23E48"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_DISPLAY = "Fraunces, Georgia, serif"
FONT_UI = "Inter, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SF Mono', Menlo, monospace"

# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_UI, color=INK, size=13),
        title=dict(font=dict(family=FONT_DISPLAY, size=18, color=INK), x=0.0),
        colorway=CATEGORICAL,
        xaxis=dict(
            gridcolor=LINE_FAINT, linecolor=LINE_SOFT, zerolinecolor=LINE_SOFT,
            tickfont=dict(family=FONT_MONO, size=11, color=INK_MUTED),
            title_font=dict(size=12, color=INK_MUTED),
        ),
        yaxis=dict(
            gridcolor=LINE_FAINT, linecolor=LINE_SOFT, zerolinecolor=LINE_SOFT,
            tickfont=dict(family=FONT_MONO, size=11, color=INK_MUTED),
            title_font=dict(size=12, color=INK_MUTED),
        ),
        legend=dict(
            font=dict(size=12, color=INK_MUTED),
            bgcolor="rgba(0,0,0,0)", orientation="h",
            yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        hoverlabel=dict(
            bgcolor=INK, bordercolor=INK,
            font=dict(family=FONT_MONO, size=12, color=PAPER),
        ),
        margin=dict(l=48, r=24, t=48, b=44),
    )
)
pio.templates["pae"] = _template
pio.templates.default = "pae"


def base_figure(height: int = 380) -> go.Figure:
    """A figure pre-wired to the design system template."""
    fig = go.Figure()
    fig.update_layout(template="pae", height=height)
    return fig


# ---------------------------------------------------------------------------
# Page chrome (CSS injected once per page render)
# ---------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{ background-color: {PAPER}; }}
[data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid {LINE_SOFT}; }}
[data-testid="stSidebar"] * {{ color: {INK}; }}

h1, h2, h3 {{ font-family: {FONT_DISPLAY} !important; color: {INK} !important; letter-spacing: -0.01em; }}
p, li, label, .stMarkdown {{ font-family: {FONT_UI}; color: {INK}; }}

/* headline block */
.pae-kicker {{
  font-family: {FONT_MONO}; font-size: 0.72rem; letter-spacing: 0.18em;
  text-transform: uppercase; color: {VERMILION}; margin-bottom: 0.15rem;
}}
.pae-title {{
  font-family: {FONT_DISPLAY}; font-size: 2.1rem; font-weight: 600;
  color: {INK}; line-height: 1.1; margin: 0 0 0.3rem 0;
}}
.pae-dek {{
  font-family: {FONT_UI}; font-size: 0.95rem; color: {INK_MUTED};
  max-width: 46rem; margin-bottom: 1.2rem;
}}

/* KPI tiles */
.pae-tiles {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 0.8rem; }}
.pae-tile {{
  flex: 1 1 130px; background: {PANEL}; border: 1px solid {LINE_SOFT};
  border-radius: 10px; padding: 12px 14px 10px 14px; min-width: 130px;
}}
.pae-tile .k {{
  font-family: {FONT_UI}; font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase; color: {INK_MUTED};
}}
.pae-tile .v {{
  font-family: {FONT_MONO}; font-size: 1.55rem; font-weight: 600;
  color: {INK}; line-height: 1.25;
}}
.pae-tile .d {{ font-family: {FONT_MONO}; font-size: 0.75rem; }}
.pae-up {{ color: {GOOD}; }}
.pae-down {{ color: {SERIOUS}; }}
.pae-flat {{ color: {INK_FAINT}; }}

/* verdict / callout card */
.pae-verdict {{
  border-left: 4px solid {VERMILION}; background: {PANEL};
  border-radius: 0 10px 10px 0; padding: 14px 18px; margin: 0.6rem 0 1rem 0;
  font-family: {FONT_UI}; color: {INK};
}}
.pae-verdict .h {{ font-family: {FONT_DISPLAY}; font-size: 1.1rem; font-weight: 600; margin-bottom: 4px; }}
.pae-verdict .mono {{ font-family: {FONT_MONO}; }}

/* section rule */
.pae-rule {{ border: 0; border-top: 1px solid {LINE_SOFT}; margin: 1.4rem 0 1rem 0; }}
</style>
"""


def inject_css(st) -> None:
    """Inject the design-system CSS into the current Streamlit page."""
    st.markdown(CSS, unsafe_allow_html=True)


def headline(st, kicker: str, title: str, dek: str) -> None:
    """Render the editorial headline block: kicker, display title, dek."""
    st.markdown(
        f'<div class="pae-kicker">{kicker}</div>'
        f'<div class="pae-title">{title}</div>'
        f'<div class="pae-dek">{dek}</div>',
        unsafe_allow_html=True,
    )


def kpi_tiles(st, tiles: list[dict]) -> None:
    """Render a row of KPI tiles.

    Each tile dict: {"label": str, "value": str, "delta": str | None,
    "direction": "up" | "down" | "flat" | None}.
    Direction colors the delta text; "up" is always styled as good, so pass
    the semantic direction (a falling bounce rate should be "up").
    """
    parts = ['<div class="pae-tiles">']
    for t in tiles:
        delta_html = ""
        if t.get("delta"):
            cls = {"up": "pae-up", "down": "pae-down"}.get(t.get("direction") or "", "pae-flat")
            arrow = {"up": "&#9650;", "down": "&#9660;"}.get(t.get("direction") or "", "&#8212;")
            delta_html = f'<div class="d {cls}">{arrow} {t["delta"]}</div>'
        parts.append(
            f'<div class="pae-tile"><div class="k">{t["label"]}</div>'
            f'<div class="v">{t["value"]}</div>{delta_html}</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def verdict_card(st, heading: str, body_html: str, color: str = VERMILION) -> None:
    """Render a verdict/callout card with a colored spine."""
    st.markdown(
        f'<div class="pae-verdict" style="border-left-color:{color}">'
        f'<div class="h">{heading}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def rule(st) -> None:
    """A quiet horizontal rule between page sections."""
    st.markdown('<hr class="pae-rule" />', unsafe_allow_html=True)


def fmt_int(n: float) -> str:
    """Format a count with thousands separators (12,847)."""
    return f"{int(round(n)):,}"


def fmt_compact(n: float) -> str:
    """Compact human format: 1.2M, 84.3K, 912."""
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,.0f}"


def fmt_money(n: float) -> str:
    """Compact currency format: $1.2M, $84.3K, $912."""
    return "$" + fmt_compact(n)


def fmt_duration(seconds: float) -> str:
    """Format seconds as m:ss."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"
