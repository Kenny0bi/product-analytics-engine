"""Shared pytest fixtures for the Product Analytics Engine test suite.

Provides a small-scale event dataset (100 users, ~2000 events) for unit tests
and a pre-loaded DuckDB connection for integration tests.
"""

from pathlib import Path

import duckdb
import pytest

from src.config.settings import Settings
from src.data.event_generator import EventGenerator
from src.data.ingestion import DuckDBIngestion


@pytest.fixture(scope="session")
def small_settings(tmp_path_factory) -> Settings:
    """Settings configured for a small test dataset."""
    tmp_dir = tmp_path_factory.mktemp("test_data")
    settings = Settings()
    settings.num_users = 100
    settings.target_events = 2000
    settings.random_seed = 42
    settings.data_dir = tmp_dir
    settings.db_path = tmp_dir / "test_analytics.duckdb"
    return settings


@pytest.fixture(scope="session")
def event_generator(small_settings) -> EventGenerator:
    """An EventGenerator configured for the small test dataset."""
    return EventGenerator(small_settings)


@pytest.fixture(scope="session")
def generated_data(event_generator, small_settings):
    """Generate the full small dataset: users, sessions, events, experiments."""
    users = event_generator.generate_users()
    sessions = event_generator.generate_sessions(users)
    events = event_generator.generate_events(sessions, users)
    experiments, assignments = event_generator.generate_experiments(users, events)
    return {
        "users": users,
        "sessions": sessions,
        "events": events,
        "experiments": experiments,
        "assignments": assignments,
        "settings": small_settings,
    }


@pytest.fixture(scope="session")
def test_db(generated_data) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection loaded with the small test dataset.

    Returns a persistent connection that survives the full test session.
    The database file is written to a temporary directory and cleaned
    up automatically by pytest.
    """
    settings = generated_data["settings"]
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    ingestion = DuckDBIngestion(settings.db_path)
    ingestion.initialize_schema()
    ingestion.load_users(generated_data["users"])
    ingestion.load_sessions(generated_data["sessions"])
    ingestion.load_events(generated_data["events"])
    ingestion.load_experiments(
        generated_data["experiments"], generated_data["assignments"]
    )
    # Materialize daily metrics the same way the Dagster pipeline does, so
    # forecast and anomaly endpoints have a populated daily_metrics table.
    ingestion.conn.execute("""
        insert into daily_metrics
        select cast(timestamp as date), 'dau', count(distinct user_id),
               'overall', 'overall'
        from events group by 1
        union all
        select cast(timestamp as date), 'revenue', coalesce(sum(revenue), 0),
               'overall', 'overall'
        from events group by 1
        union all
        select cast(created_at as date), 'signups', count(*),
               'overall', 'overall'
        from users group by 1
    """)
    # Release the read-write lock so the API tests (which open the same file
    # read-only in-process) can connect. Read-only connections can coexist.
    ingestion.close()

    conn = duckdb.connect(settings.db_path.as_posix(), read_only=True)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def test_db_path(test_db, generated_data) -> Path:
    """Path to the test DuckDB database file.

    Depends on ``test_db`` (not just ``generated_data``) so the database
    file is actually written before API tests try to open it.
    """
    return generated_data["settings"].db_path
