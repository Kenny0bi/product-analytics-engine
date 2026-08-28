"""
Application-wide configuration using Pydantic BaseSettings.

All settings can be overridden via environment variables prefixed with PAE_
(e.g., PAE_DB_PATH, PAE_NUM_USERS). Defaults are tuned for the full 50K-user
dataset; use smaller values for testing.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for the Product Analytics Engine.

    Controls data generation parameters, session behavior, the canonical
    conversion funnel definition, and API server settings. Values are
    loaded from environment variables (PAE_ prefix) with sensible defaults
    for local development.
    """

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    db_path: Path = data_dir / "analytics.duckdb"

    # -------------------------------------------------------------------------
    # Data generation
    # -------------------------------------------------------------------------
    num_users: int = 50_000
    target_events: int = 2_000_000
    date_start: str = "2025-01-01"
    date_end: str = "2025-12-31"
    random_seed: int = 42

    # Event weights: probability of each event type within a session.
    # Must sum to 1.0. Adjusted during generation so the first event
    # in every session is always a page_view.
    event_weights: dict[str, float] = {
        "page_view": 0.40,
        "click": 0.25,
        "feature_use": 0.15,
        "signup": 0.05,
        "purchase": 0.05,
        "form_submit": 0.05,
        "search": 0.05,
    }

    # -------------------------------------------------------------------------
    # Session parameters
    # -------------------------------------------------------------------------
    avg_sessions_per_user: float = 8.0
    avg_events_per_session: float = 5.0
    session_timeout_minutes: int = 30

    # -------------------------------------------------------------------------
    # Conversion funnel definition
    # The canonical funnel used for default funnel analysis. Each entry is an
    # event_name that must be reached in order for a user to progress.
    # -------------------------------------------------------------------------
    conversion_funnel: list[str] = [
        "view_homepage",
        "view_product",
        "click_add_to_cart",
        "begin_checkout",
        "complete_purchase",
    ]

    # -------------------------------------------------------------------------
    # API server
    # -------------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {
        "env_prefix": "PAE_",
    }
