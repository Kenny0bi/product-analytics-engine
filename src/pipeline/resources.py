"""
Dagster resource for DuckDB connections.

Provides a ``ConfigurableResource`` that manages DuckDB connections for
pipeline assets.  Each asset that needs database access declares this
resource as a dependency and calls ``get_connection()`` to obtain a
connection scoped to the asset's execution.
"""

from __future__ import annotations

import duckdb
from dagster import ConfigurableResource
from loguru import logger


class DuckDBResource(ConfigurableResource):
    """Dagster-managed DuckDB connection resource.

    Parameters
    ----------
    db_path : str
        Filesystem path to the DuckDB database file
        (e.g. ``"data/analytics.duckdb"``).
    """

    db_path: str

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Open and return a DuckDB connection.

        Each call creates a new connection so that concurrent asset
        materializations do not share transaction state.

        Returns
        -------
        duckdb.DuckDBPyConnection
        """
        logger.debug("Opening DuckDB connection: {}", self.db_path)
        return duckdb.connect(self.db_path)
