"""FastAPI application for the Product Analytics Engine.

Provides REST endpoints for querying product metrics, funnels, cohort retention,
A/B experiment results, user segments, and metric forecasts. Uses DuckDB as the
analytical store, with a connection initialized at application startup via the
ASGI lifespan protocol.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import duckdb
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config.settings import Settings
from src.serving.routes import (
    experiments,
    forecasting,
    funnels,
    metrics,
    retention,
    segments,
)


def get_db(request: Request) -> duckdb.DuckDBPyConnection:
    """Retrieve a thread-local DuckDB connection from the application state.

    DuckDB connections are not thread-safe, so each request opens its own
    connection to the same database file.
    """
    return duckdb.connect(request.app.state.db_path, read_only=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize the DuckDB database path on startup and clean up on shutdown."""
    settings = Settings()
    app.state.db_path = settings.db_path.as_posix()
    logger.info("Analytics API starting, database at {}", app.state.db_path)

    # Validate that the database file is accessible
    try:
        conn = duckdb.connect(app.state.db_path, read_only=True)
        tables = conn.execute("show tables").fetchall()
        logger.info("Database contains {} tables", len(tables))
        conn.close()
    except Exception as exc:
        logger.warning("Database not ready: {}. Endpoints may return errors.", exc)

    yield

    logger.info("Analytics API shutting down")


app = FastAPI(
    title="Product Analytics Engine API",
    description=(
        "Query product metrics, funnels, cohort retention, A/B experiments, "
        "user segments, and metric forecasts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(metrics.router)
app.include_router(funnels.router)
app.include_router(retention.router)
app.include_router(experiments.router)
app.include_router(segments.router)
app.include_router(forecasting.router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe for container orchestration."""
    return {"status": "ok"}
