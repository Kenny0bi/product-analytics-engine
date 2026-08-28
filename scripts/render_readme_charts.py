"""Render the chart images embedded in README.md from the live database.

Usage:
    pip install kaleido
    python scripts/render_readme_charts.py

Writes PNGs to docs/charts/. Requires data/analytics.duckdb to exist
(run `python -m src.main generate` and `python -m src.main analyze` first).
The images in the README are real renders of the real dataset, so this
script is how they get refreshed when the generator or the charts change.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import plotly.io as pio  # noqa: E402

from src.analytics.funnels import FunnelAnalyzer  # noqa: E402
from src.analytics.retention import RetentionAnalyzer  # noqa: E402
from src.config.settings import Settings  # noqa: E402
from src.dashboard import theme as T  # noqa: E402
from src.dashboard.pages import experiments as ex  # noqa: E402
from src.dashboard.pages import forecasting as fo  # noqa: E402
from src.dashboard.pages import funnels as fu  # noqa: E402
from src.dashboard.pages import overview as ov  # noqa: E402
from src.dashboard.pages import retention as re_  # noqa: E402
from src.dashboard.pages import segments as sg  # noqa: E402
from src.statistics.ab_testing import ABTestAnalyzer  # noqa: E402
from src.statistics.forecasting import MetricForecaster  # noqa: E402
from src.statistics.sequential import SequentialTester  # noqa: E402

OUT = ROOT / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str, width: int = 1200) -> None:
    """Write one chart as PNG on the paper surface."""
    fig.update_layout(paper_bgcolor=T.PAPER, plot_bgcolor=T.PAPER)
    # Kaleido v2 serializes with orjson, which cannot handle pandas
    # Timestamps; a JSON round-trip through plotly's encoder fixes that.
    fig = pio.from_json(pio.to_json(fig))
    fig.write_image(str(OUT / f"{name}.png"), width=width, scale=1.5)
    print(f"wrote docs/charts/{name}.png")


def main() -> None:
    conn = duckdb.connect(Settings().db_path.as_posix(), read_only=True)

    save(ov._pulse_chart(ov._weekly_pulse(conn)), "pulse", 1100)

    fa = FunnelAnalyzer(conn)
    res = fa.compute_funnel(fu._DEFAULT_STEPS)
    save(fu._river_chart(fu._DEFAULT_STEPS, [s.users for s in res.steps]),
         "river", 1300)

    matrix = RetentionAnalyzer(conn).compute_retention("month", 12)
    save(re_._comet_chart(matrix, annotate=True), "comets", 1200)

    ab = ABTestAnalyzer(conn)
    save(ex._posterior_chart(
        ab._fetch_variant_stats("checkout_flow_redesign"), "binary"),
        "posterior", 1000)
    sprt = SequentialTester(conn).compute_sprt("checkout_flow_redesign")
    save(ex._sprt_chart(conn, "checkout_flow_redesign", sprt), "sprt", 1100)

    clustered = sg._clusters.__wrapped__(conn, 5)  # bypass st.cache_data
    save(sg._fingerprint_chart(clustered), "fingerprints", 1300)

    fc = MetricForecaster(conn).forecast_metric("dau", periods_ahead=30)
    save(fo._cone_chart(fo._history(conn, "dau"), fc, False, False),
         "cone", 1200)

    conn.close()


if __name__ == "__main__":
    main()
