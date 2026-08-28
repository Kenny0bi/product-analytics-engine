"""CLI entry point for data generation and ingestion.

Usage::

    python -m src.data.cli generate --num-users 50000 --seed 42

Generates synthetic users, sessions, events, and experiments, then loads
everything into the DuckDB analytical store and runs validation checks.
"""

from __future__ import annotations

import click
from loguru import logger

from src.config.settings import Settings


@click.group()
def data_cli() -> None:
    """Data generation and loading commands."""


@data_cli.command()
@click.option(
    "--num-users",
    default=50_000,
    show_default=True,
    help="Number of synthetic users to generate.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
def generate(num_users: int, seed: int) -> None:
    """Generate synthetic event data and load into DuckDB.

    Produces users, sessions, in-session events, and three A/B experiments
    with known ground-truth effects.  Data is inserted into the DuckDB
    database at the configured ``db_path`` and validated post-load.
    """
    # Defer heavy imports so CLI help is instant.
    from src.data.event_generator import EventGenerator
    from src.data.ingestion import DuckDBIngestion

    settings = Settings(num_users=num_users, random_seed=seed)
    logger.info(
        "Starting data generation: {} users, seed={}",
        num_users,
        seed,
    )

    # Generate users and sessions in memory, then stream events straight
    # into DuckDB in chunks: a full year of events is millions of rows, and
    # holding them all in Python before loading needs more RAM than the
    # laptop this runs on has.
    gen = EventGenerator(settings)
    users = gen.generate_users()
    sessions = gen.generate_sessions(users)

    ingestion = DuckDBIngestion(settings.db_path)
    ingestion.initialize_schema()
    events = gen.generate_events(sessions, users, sink=ingestion.load_events)
    experiments_df, assignments_df = gen.generate_experiments(users, events)

    # Sessions and users are loaded after event generation because that
    # pass back-fills session aggregates and user lifetime stats.
    ingestion.load_users(users)
    ingestion.load_sessions(sessions)
    ingestion.load_experiments(experiments_df, assignments_df)

    # 3. Validate
    results = ingestion.validate_load()
    logger.info("Validation results: {}", results)

    if results["all_checks_passed"]:
        logger.info("Data generation complete. Database: {}", settings.db_path)
    else:
        logger.error("Validation failures detected -- inspect results above")
        raise click.Abort()


if __name__ == "__main__":
    data_cli()
