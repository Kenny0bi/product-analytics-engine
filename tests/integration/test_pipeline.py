"""Integration tests for the pipeline layer.

Covers the Dagster definitions loading (which catches asset signature and
dependency-graph errors), the one-shot daily_metrics backfill, and the
materialized view refresh. The write-path tests copy the read-only test
database first, because DuckDB does not allow a read-write connection on
a file that in-process read-only connections already hold.
"""

import shutil

import duckdb
import pytest

from src.pipeline.backfill import backfill_daily_metrics, refresh_materialized_views


@pytest.fixture()
def writable_db(test_db_path, tmp_path):
    """A read-write DuckDB connection on a private copy of the test data."""
    copy_path = tmp_path / "pipeline_test.duckdb"
    shutil.copy(test_db_path, copy_path)
    conn = duckdb.connect(copy_path.as_posix())
    yield conn
    conn.close()


def test_dagster_definitions_load() -> None:
    """The Dagster asset graph must build: nine assets, two schedules."""
    from src.pipeline.definitions import defs

    asset_keys = list(defs.resolve_asset_graph().get_all_asset_keys())
    assert len(asset_keys) == 9, f"Expected 9 assets, got {len(asset_keys)}"
    assert len(list(defs.schedules)) == 2


def test_backfill_daily_metrics_produces_rows(writable_db) -> None:
    """The backfill must write every core metric for every active day."""
    rows = backfill_daily_metrics(writable_db)
    assert rows > 0

    metrics = {r[0] for r in writable_db.execute(
        "select distinct metric_name from daily_metrics").fetchall()}
    for expected in ("dau", "revenue", "signups", "conversion_rate",
                     "avg_session_duration", "bounce_rate"):
        assert expected in metrics, f"Missing metric: {expected}"

    # Idempotency: running twice must not duplicate rows.
    again = backfill_daily_metrics(writable_db)
    assert again == rows


def test_materialized_views_are_created(writable_db) -> None:
    """The refresh must build all three pre-aggregated tables, populated."""
    refresh_materialized_views(writable_db)
    for table in ("mv_funnel_daily", "mv_session_daily", "mv_user_segments"):
        count = writable_db.execute(f"select count(*) from {table}").fetchone()[0]
        assert count > 0, f"{table} is empty"
