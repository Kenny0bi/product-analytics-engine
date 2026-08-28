"""Product Analytics Engine -- main CLI entry point.

Provides four commands for the full lifecycle:
  generate   Generate synthetic event data and load into DuckDB.
  analyze    Run all analytics computations (funnels, retention, segments, experiments).
  serve      Start the FastAPI REST API server.
  dashboard  Start the Streamlit dashboard.

Usage:
    python -m src.main generate --num-users 50000 --seed 42
    python -m src.main analyze
    python -m src.main serve
    python -m src.main dashboard
"""

import subprocess
import sys
from pathlib import Path

import click
from loguru import logger

from src.config.settings import Settings

# Configure loguru: remove default handler, add one with structured format
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Product Analytics Engine CLI."""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")


@cli.command()
@click.option("--num-users", default=50_000, show_default=True, help="Number of users to generate.")
@click.option("--target-events", default=2_000_000, show_default=True, help="Target event count.")
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducibility.")
@click.option("--db-path", default=None, help="Override DuckDB path.")
def generate(num_users: int, target_events: int, seed: int, db_path: str | None) -> None:
    """Generate synthetic event data and load into DuckDB.

    Creates users, sessions, events, and experiments using the configured
    distributions, then ingests everything into the analytical store.
    """
    from src.data.event_generator import EventGenerator
    from src.data.ingestion import DuckDBIngestion

    settings = Settings()
    settings.num_users = num_users
    settings.target_events = target_events
    settings.random_seed = seed

    if db_path:
        settings.db_path = Path(db_path)

    settings.data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating {} users with seed {}", num_users, seed)
    gen = EventGenerator(settings)

    logger.info("Generating user profiles...")
    users = gen.generate_users()
    logger.info("Generated {} users", len(users))

    logger.info("Generating sessions...")
    sessions = gen.generate_sessions(users)
    logger.info("Generated {} sessions", len(sessions))

    logger.info("Generating events (streamed into DuckDB in chunks)...")
    ingestion = DuckDBIngestion(settings.db_path)
    ingestion.initialize_schema()
    events = gen.generate_events(sessions, users, sink=ingestion.load_events)

    logger.info("Generating experiments...")
    experiments, assignments = gen.generate_experiments(users, events)
    logger.info("Generated {} experiments with {} assignments", len(experiments), len(assignments))

    logger.info("Loading users, sessions, and experiments into {}", settings.db_path)
    ingestion.load_users(users)
    ingestion.load_sessions(sessions)
    ingestion.load_experiments(experiments, assignments)

    validation = ingestion.validate_load()
    logger.info("Validation results: {}", validation)

    logger.info("Data generation complete.")


@cli.command()
@click.option("--db-path", default=None, help="Override DuckDB path.")
def analyze(db_path: str | None) -> None:
    """Run all analytics computations on the loaded data.

    Computes daily metrics, funnel conversion, cohort retention, RFM segments,
    behavioral clusters, and experiment analyses.
    """
    import duckdb

    from src.analytics.funnels import FunnelAnalyzer
    from src.analytics.retention import RetentionAnalyzer
    from src.analytics.segmentation import SegmentationAnalyzer
    from src.analytics.sessions import SessionAnalyzer
    from src.pipeline.backfill import (
        backfill_daily_metrics,
        refresh_materialized_views,
    )
    from src.statistics.ab_testing import ABTestAnalyzer

    settings = Settings()
    if db_path:
        settings.db_path = Path(db_path)

    conn = duckdb.connect(settings.db_path.as_posix())

    logger.info("Materializing daily metrics...")
    backfill_daily_metrics(conn)

    logger.info("Refreshing materialized views...")
    refresh_materialized_views(conn)

    logger.info("Computing session metrics...")
    session_analyzer = SessionAnalyzer(conn)
    session_metrics = session_analyzer.compute_session_metrics()
    logger.info("Session metrics: {} total sessions", session_metrics.total_sessions)

    logger.info("Computing funnel conversion...")
    funnel_analyzer = FunnelAnalyzer(conn)
    funnel = funnel_analyzer.compute_funnel(settings.conversion_funnel)
    logger.info(
        "Funnel overall conversion: {:.2%}", funnel.overall_conversion_rate
    )

    logger.info("Computing cohort retention...")
    retention_analyzer = RetentionAnalyzer(conn)
    retention = retention_analyzer.compute_retention(period="month", num_periods=12)
    logger.info("Retention computed for {} cohorts", len(retention.cohorts))

    logger.info("Computing RFM segmentation...")
    seg_analyzer = SegmentationAnalyzer(conn)
    rfm = seg_analyzer.compute_rfm()
    logger.info("RFM segments: {}", rfm["rfm_segment"].value_counts().to_dict())

    logger.info("Computing behavioral clusters...")
    clusters = seg_analyzer.behavioral_clustering()
    logger.info("Clusters: {}", clusters["cluster_label"].value_counts().to_dict())

    logger.info("Analyzing experiments...")
    ab_analyzer = ABTestAnalyzer(conn)
    experiments = conn.execute("select experiment_id from experiments").fetchall()
    for (exp_id,) in experiments:
        result = ab_analyzer.analyze_experiment(exp_id)
        logger.info(
            "Experiment {}: lift={:.1%}, p={:.4f}, P(B>A)={:.3f}",
            exp_id, result.observed_lift,
            result.frequentist.p_value,
            result.bayesian.prob_treatment_better,
        )

    conn.close()
    logger.info("All analytics computations complete.")


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="API host.")
@click.option("--port", default=8000, show_default=True, help="API port.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI REST API server."""
    import uvicorn

    logger.info("Starting API server on {}:{}", host, port)
    uvicorn.run(
        "src.serving.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option("--port", default=8501, show_default=True, help="Streamlit port.")
def dashboard(port: int) -> None:
    """Start the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    logger.info("Starting Streamlit dashboard on port {}", port)
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
        ],
        check=True,
    )


if __name__ == "__main__":
    cli()
