"""Streamlit entry point for the Product Analytics Engine dashboard.

Dispatches to six page modules (Overview, Funnels, Retention, Experiments,
Segments, Forecasting), each of which renders its own visualizations from
live DuckDB queries. There is deliberately no demo-data fallback: if the
database is missing, the dashboard says so and tells you how to build it,
instead of showing fabricated numbers styled as real ones.
"""

import duckdb
import streamlit as st
from loguru import logger

from src.config.settings import Settings
from src.dashboard import theme as T
from src.dashboard.pages import (
    experiments,
    forecasting,
    funnels,
    overview,
    retention,
    segments,
)

_PAGES = {
    "Overview": overview,
    "Funnels": funnels,
    "Retention": retention,
    "Experiments": experiments,
    "Segments": segments,
    "Forecasting": forecasting,
}


def _get_connection() -> duckdb.DuckDBPyConnection | None:
    """Open a read-only DuckDB connection, or None if the database is missing."""
    db_path = Settings().db_path
    if not db_path.exists():
        logger.warning("Database not found at {}", db_path)
        return None
    try:
        return duckdb.connect(db_path.as_posix(), read_only=True)
    except Exception as exc:
        logger.warning("Could not connect to database: {}", exc)
        return None


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="Product Analytics Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    T.inject_css(st)

    st.sidebar.markdown(
        f'<div style="font-family:{T.FONT_DISPLAY};font-size:1.35rem;'
        f'font-weight:600;line-height:1.2;margin-bottom:0.1rem;">'
        f'Product Analytics Engine</div>'
        f'<div style="font-family:{T.FONT_MONO};font-size:0.7rem;'
        f'letter-spacing:0.14em;text-transform:uppercase;color:{T.VERMILION};'
        f'margin-bottom:1rem;">50K users &middot; 2M events &middot; DuckDB</div>',
        unsafe_allow_html=True,
    )

    page_name = st.sidebar.radio("Navigation", list(_PAGES.keys()),
                                 label_visibility="collapsed")

    st.sidebar.markdown(
        f'<div style="margin-top:2rem;font-size:0.78rem;color:{T.INK_MUTED};">'
        'Synthetic event stream &#8594; DuckDB &#8594; funnels, cohorts, '
        'segments, A/B statistics, forecasts.<br><br>'
        'Every number on every page is computed live from SQL and the '
        'analytics modules; nothing is precomputed for display.</div>',
        unsafe_allow_html=True,
    )

    conn = _get_connection()
    try:
        _PAGES[page_name].render(conn)
    except Exception as exc:
        st.error(f"Error rendering {page_name}: {exc}")
        logger.exception("Dashboard page {} failed", page_name)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
