"""DuckDB data ingestion with schema enforcement and validation.

Loads generated DataFrames (users, events, sessions, experiments) into
DuckDB tables defined in ``sql/schema.sql``.  Each load method validates
basic constraints before insertion and returns the inserted row count.
``validate_load`` runs post-hoc referential integrity, uniqueness, and
business-rule checks against the populated database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from loguru import logger


class DuckDBIngestion:
    """Load generated DataFrames into DuckDB with schema enforcement.

    Parameters
    ----------
    db_path : Path
        Filesystem path for the DuckDB database file. Created if absent.
    """

    _ALLOWED_EVENT_TYPES = frozenset(
        ["page_view", "click", "signup", "purchase", "feature_use", "form_submit", "search"]
    )

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        logger.info("Connected to DuckDB at {}", db_path)

    def close(self) -> None:
        """Close the underlying DuckDB connection, releasing the file lock.

        DuckDB holds an exclusive lock while a read-write connection is open,
        so any process (or the API server) that wants to open the same file
        read-only must wait until this is called.
        """
        self.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Create all tables from ``sql/schema.sql``, dropping existing ones first.

        This ensures a clean slate for data loading and avoids schema-drift
        issues during iterative development.
        """
        schema_path = Path(__file__).resolve().parent.parent.parent / "sql" / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        # Drop in dependency order.
        for table in [
            "daily_metrics",
            "experiment_assignments",
            "experiments",
            "sessions",
            "events",
            "users",
        ]:
            self.conn.execute(f"drop table if exists {table}")

        sql_text = schema_path.read_text()
        # Strip SQL single-line comments before splitting on semicolons,
        # because comments may contain literal semicolons that would
        # incorrectly split a CREATE TABLE statement.
        import re
        sql_no_comments = re.sub(r"--[^\n]*", "", sql_text)
        for statement in sql_no_comments.split(";"):
            stmt = statement.strip()
            if not stmt:
                continue
            self.conn.execute(stmt)

        logger.info("Schema initialized from {}", schema_path)

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def load_users(self, df: pd.DataFrame) -> int:
        """Insert user rows after validating no null user_ids.

        Returns the number of rows inserted.
        """
        null_count = df["user_id"].isna().sum()
        if null_count:
            raise ValueError(f"Found {null_count} null user_ids in users DataFrame")

        self.conn.execute("insert into users select * from df")
        count = self.conn.execute("select count(*) from users").fetchone()[0]
        logger.info("Loaded {} users", count)
        return count

    def load_events(self, df: pd.DataFrame, batch_size: int = 50_000) -> int:
        """Insert events in batches after validating event_type values.

        Batch insertion avoids memory pressure for large event DataFrames
        (2M+ rows).  Returns total rows inserted.
        """
        invalid = set(df["event_type"].unique()) - self._ALLOWED_EVENT_TYPES
        if invalid:
            raise ValueError(f"Invalid event_type values: {invalid}")

        total = 0
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            self.conn.execute("insert into events select * from batch")
            total += len(batch)
            logger.debug("Loaded event batch {}-{}", start, start + len(batch))

        logger.info("Loaded {} events total", total)
        return total

    def load_sessions(self, df: pd.DataFrame) -> int:
        """Insert session rows. Returns the number of rows inserted.

        Selects only columns present in the sessions table schema, ignoring
        extra columns the generator may carry (e.g. browser, os which live
        on the events table).
        """
        session_cols = [
            "session_id", "user_id", "started_at", "ended_at",
            "duration_seconds", "event_count", "page_view_count",
            "has_conversion", "entry_page", "exit_page",
            "device_type", "utm_source", "utm_medium", "utm_campaign",
        ]
        sessions_clean = df[session_cols]  # noqa: F841  (read by DuckDB replacement scan)
        self.conn.execute("insert into sessions select * from sessions_clean")
        count = self.conn.execute("select count(*) from sessions").fetchone()[0]
        logger.info("Loaded {} sessions", count)
        return count

    def load_experiments(
        self, experiments_df: pd.DataFrame, assignments_df: pd.DataFrame
    ) -> None:
        """Insert experiment definitions and user-to-variant assignments."""
        self.conn.execute("insert into experiments select * from experiments_df")
        self.conn.execute(
            "insert into experiment_assignments select * from assignments_df"
        )
        exp_count = self.conn.execute("select count(*) from experiments").fetchone()[0]
        assign_count = self.conn.execute(
            "select count(*) from experiment_assignments"
        ).fetchone()[0]
        logger.info(
            "Loaded {} experiments and {} assignments", exp_count, assign_count
        )

    # ------------------------------------------------------------------
    # Post-load validation
    # ------------------------------------------------------------------

    def validate_load(self) -> dict[str, Any]:
        """Run post-load integrity checks and return a results dict.

        Checks performed
        ----------------
        1. Referential integrity: every ``events.user_id`` exists in ``users``.
        2. Referential integrity: every ``events.session_id`` exists in ``sessions``.
        3. No null timestamps in the events table.
        4. No duplicate ``event_id`` values.
        5. All purchase events have ``revenue > 0``.
        6. ``session.started_at <= session.ended_at`` for all sessions.
        7. Row counts reported for users, events, and sessions.
        """
        results: dict[str, Any] = {}

        # Row counts
        results["user_count"] = self.conn.execute(
            "select count(*) from users"
        ).fetchone()[0]
        results["event_count"] = self.conn.execute(
            "select count(*) from events"
        ).fetchone()[0]
        results["session_count"] = self.conn.execute(
            "select count(*) from sessions"
        ).fetchone()[0]

        # 1. Orphan events (user_id not in users)
        orphan_users = self.conn.execute("""
            select count(*) from events e
            where not exists (select 1 from users u where u.user_id = e.user_id)
        """).fetchone()[0]
        results["orphan_event_users"] = orphan_users

        # 2. Orphan events (session_id not in sessions)
        orphan_sessions = self.conn.execute("""
            select count(*) from events e
            where not exists (select 1 from sessions s where s.session_id = e.session_id)
        """).fetchone()[0]
        results["orphan_event_sessions"] = orphan_sessions

        # 3. Null timestamps
        null_ts = self.conn.execute(
            "select count(*) from events where timestamp is null"
        ).fetchone()[0]
        results["null_timestamps"] = null_ts

        # 4. Duplicate event_ids
        dup_events = self.conn.execute("""
            select count(*) from (
                select event_id from events group by event_id having count(*) > 1
            )
        """).fetchone()[0]
        results["duplicate_event_ids"] = dup_events

        # 5. Purchase events without revenue
        bad_purchases = self.conn.execute("""
            select count(*) from events
            where event_name = 'complete_purchase'
              and (revenue is null or revenue <= 0)
        """).fetchone()[0]
        results["purchases_without_revenue"] = bad_purchases

        # 6. Session temporal ordering
        bad_sessions = self.conn.execute("""
            select count(*) from sessions
            where ended_at is not null and started_at > ended_at
        """).fetchone()[0]
        results["sessions_start_after_end"] = bad_sessions

        # Summary
        all_ok = all([
            orphan_users == 0,
            orphan_sessions == 0,
            null_ts == 0,
            dup_events == 0,
            bad_purchases == 0,
            bad_sessions == 0,
        ])
        results["all_checks_passed"] = all_ok

        if all_ok:
            logger.info("All validation checks passed")
        else:
            logger.warning("Validation issues found: {}", results)

        return results
