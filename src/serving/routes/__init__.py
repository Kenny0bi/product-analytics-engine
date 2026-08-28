"""API route modules for the Product Analytics Engine.

Each module defines an APIRouter with endpoints for a specific analytics domain:
metrics, funnels, retention, experiments, segments, and forecasting.
"""

from src.serving.routes import (
    experiments,
    forecasting,
    funnels,
    metrics,
    retention,
    segments,
)

__all__ = [
    "experiments",
    "forecasting",
    "funnels",
    "metrics",
    "retention",
    "segments",
]
