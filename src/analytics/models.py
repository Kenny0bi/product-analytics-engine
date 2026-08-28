"""Pydantic models for analytics computation results.

Each model represents the output of one analytics engine.  They are used as
return types in the funnel, retention, segmentation, and session analyzers,
and as FastAPI response schemas in the serving layer.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Funnel analysis
# ---------------------------------------------------------------------------

class FunnelStep(BaseModel):
    """One step in a conversion funnel."""

    step_name: str
    step_index: int
    users: int = Field(description="Distinct users who reached this step")
    conversion_rate: float = Field(
        description="Fraction of users from the previous step who reached this one"
    )
    overall_conversion_rate: float = Field(
        description="Fraction of step-0 users who reached this step"
    )
    dropoff_count: int = Field(
        description="Users who completed the previous step but not this one"
    )
    dropoff_rate: float = Field(
        description="Fraction of previous-step users who dropped off"
    )
    median_time_to_next: float | None = Field(
        default=None,
        description="Median seconds between this step and the next; None for the last step",
    )


class FunnelResult(BaseModel):
    """Complete funnel analysis output."""

    funnel_name: str
    steps: list[FunnelStep]
    total_entered: int = Field(description="Users who completed step 0")
    total_converted: int = Field(description="Users who completed the final step")
    overall_conversion_rate: float
    computed_at: datetime


# ---------------------------------------------------------------------------
# Retention analysis
# ---------------------------------------------------------------------------

class RetentionMatrix(BaseModel):
    """Cohort retention matrix (weekly or monthly)."""

    period_type: str = Field(description="'week' or 'month'")
    cohorts: list[str] = Field(description="Cohort labels (e.g. '2025-01')")
    periods: list[int] = Field(description="Period offsets from signup [0, 1, ..., N]")
    cohort_sizes: dict[str, int] = Field(
        description="Cohort label -> number of users in that cohort"
    )
    retention_rates: dict[str, list[float]] = Field(
        description="Cohort label -> list of retention rates per period"
    )
    computed_at: datetime


# ---------------------------------------------------------------------------
# Session analysis
# ---------------------------------------------------------------------------

class SessionMetrics(BaseModel):
    """Aggregate session-level metrics, optionally broken down by dimension."""

    total_sessions: int
    avg_duration_seconds: float
    median_duration_seconds: float
    avg_page_depth: float = Field(
        description="Average page_view_count per session"
    )
    bounce_rate: float = Field(
        description="Fraction of sessions with 1 page view and no other events"
    )
    conversion_rate: float = Field(
        description="Fraction of sessions with has_conversion = true"
    )
    breakdowns: dict[str, dict[str, float]] | None = Field(
        default=None,
        description="Dimensional breakdown: {dimension_value: {metric: value}}",
    )
    computed_at: datetime


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

class RFMProfile(BaseModel):
    """Aggregate profile for one RFM segment."""

    segment: str
    user_count: int
    avg_recency_days: float
    avg_frequency: float
    avg_monetary: float
    pct_of_total: float = Field(
        description="Percentage of all users in this segment"
    )


class ClusterProfile(BaseModel):
    """Profile for one behavioral cluster."""

    cluster_id: int
    cluster_label: str
    user_count: int
    avg_total_sessions: float
    avg_total_events: float
    avg_total_revenue: float
    avg_session_duration: float
    avg_unique_features: float
    silhouette_score: float | None = None


class EngagementScore(BaseModel):
    """Per-user engagement score with component breakdown."""

    user_id: str
    score: float = Field(ge=0, le=100, description="Composite engagement score 0-100")
    frequency_component: float
    depth_component: float
    breadth_component: float
    recency_component: float
