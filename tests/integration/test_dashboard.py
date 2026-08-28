"""Integration tests for the Streamlit dashboard.

Runs every page headlessly through Streamlit's AppTest harness against the
small test database. A page "passes" if it renders without raising and
without surfacing a st.error/st.exception element, which catches broken
queries, chart-construction errors, and bad column references that unit
tests on the analytics layer cannot see.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest resolves relative paths against this test file, not the CWD.
_APP_PATH = str(Path(__file__).resolve().parents[2] / "src" / "dashboard" / "app.py")

_PAGES = ["Overview", "Funnels", "Retention", "Experiments",
          "Segments", "Forecasting"]


@pytest.fixture(scope="module", autouse=True)
def _point_dashboard_at_test_db(test_db_path):
    """Route the dashboard's Settings at the populated test database."""
    old = os.environ.get("PAE_DB_PATH")
    os.environ["PAE_DB_PATH"] = str(test_db_path)
    yield
    if old is None:
        os.environ.pop("PAE_DB_PATH", None)
    else:
        os.environ["PAE_DB_PATH"] = old


@pytest.mark.parametrize("page", _PAGES)
def test_dashboard_page_renders(page: str) -> None:
    """Each dashboard page must render without exceptions or error boxes."""
    at = AppTest.from_file(_APP_PATH, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()

    exceptions = [e.value for e in at.exception]
    assert not exceptions, f"{page} raised: {exceptions}"
    errors = [e.value for e in at.error]
    assert not errors, f"{page} showed an error box: {errors}"
