"""Segment endpoints — RFM analysis, behavioral clusters, and user listings.

Provides access to pre-computed user segmentation via RFM scoring and
K-means behavioral clustering, plus per-segment user enumeration.
"""

from datetime import datetime

import duckdb
from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/segments", tags=["segments"])


class RFMSegmentProfile(BaseModel):
    """Aggregate profile for one RFM segment."""

    segment: str
    user_count: int
    avg_recency_days: float
    avg_frequency: float
    avg_monetary: float
    pct_of_total: float


class RFMSummary(BaseModel):
    """Full RFM segmentation result."""

    segments: list[RFMSegmentProfile]
    total_users: int
    computed_at: datetime


class ClusterProfile(BaseModel):
    """Aggregate profile for one behavioral cluster."""

    cluster_id: int
    cluster_label: str
    user_count: int
    avg_sessions: float
    avg_events: float
    avg_revenue: float
    avg_session_duration: float
    pct_of_total: float


class ClusterSummary(BaseModel):
    """Full behavioral clustering result."""

    clusters: list[ClusterProfile]
    total_users: int
    computed_at: datetime


class UserProfile(BaseModel):
    """Minimal user record within a segment."""

    user_id: str
    plan_type: str | None = None
    signup_source: str | None = None
    country: str | None = None
    total_revenue: float = 0.0
    lifetime_events: int = 0


def _get_conn(request: Request) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(request.app.state.db_path, read_only=True)


@router.get("/rfm", response_model=RFMSummary)
async def get_rfm_segments(request: Request) -> RFMSummary:
    """Compute and return RFM segment distribution with aggregate profiles."""
    conn = _get_conn(request)
    try:
        from src.analytics.segmentation import SegmentationAnalyzer

        analyzer = SegmentationAnalyzer(conn)
        rfm_df = analyzer.compute_rfm()

        total = len(rfm_df)
        segments: list[RFMSegmentProfile] = []
        for seg_name, group in rfm_df.groupby("rfm_segment"):
            segments.append(
                RFMSegmentProfile(
                    segment=str(seg_name),
                    user_count=len(group),
                    avg_recency_days=float(group["recency_days"].mean()),
                    avg_frequency=float(group["frequency"].mean()),
                    avg_monetary=float(group["monetary"].mean()),
                    pct_of_total=round(len(group) / total * 100, 2) if total else 0.0,
                )
            )
        return RFMSummary(
            segments=sorted(segments, key=lambda s: s.user_count, reverse=True),
            total_users=total,
            computed_at=datetime.utcnow(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/clusters", response_model=ClusterSummary)
async def get_behavioral_clusters(request: Request) -> ClusterSummary:
    """Compute and return behavioral cluster profiles."""
    conn = _get_conn(request)
    try:
        from src.analytics.segmentation import SegmentationAnalyzer

        analyzer = SegmentationAnalyzer(conn)
        cluster_df = analyzer.behavioral_clustering()

        total = len(cluster_df)
        clusters: list[ClusterProfile] = []
        for cid, group in cluster_df.groupby("cluster_id"):
            label = group["cluster_label"].iloc[0] if "cluster_label" in group.columns else f"Cluster {cid}"
            clusters.append(
                ClusterProfile(
                    cluster_id=int(cid),
                    cluster_label=str(label),
                    user_count=len(group),
                    avg_sessions=float(group["total_sessions"].mean()) if "total_sessions" in group.columns else 0.0,
                    avg_events=float(group["total_events"].mean()) if "total_events" in group.columns else 0.0,
                    avg_revenue=float(group["total_revenue"].mean()) if "total_revenue" in group.columns else 0.0,
                    avg_session_duration=float(group["avg_session_duration"].mean()) if "avg_session_duration" in group.columns else 0.0,
                    pct_of_total=round(len(group) / total * 100, 2) if total else 0.0,
                )
            )
        return ClusterSummary(
            clusters=sorted(clusters, key=lambda c: c.user_count, reverse=True),
            total_users=total,
            computed_at=datetime.utcnow(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/{segment}/users", response_model=list[UserProfile])
async def get_segment_users(
    request: Request,
    segment: str = Path(..., description="Segment name, e.g. 'Champions', 'At Risk'."),
    limit: int = Query(100, ge=1, le=1000, description="Maximum users to return."),
) -> list[UserProfile]:
    """List users belonging to a specific RFM segment."""
    conn = _get_conn(request)
    try:
        from src.analytics.segmentation import SegmentationAnalyzer

        analyzer = SegmentationAnalyzer(conn)
        rfm_df = analyzer.compute_rfm()
        segment_users = rfm_df[rfm_df["rfm_segment"] == segment].head(limit)

        user_ids = segment_users["user_id"].tolist()
        if not user_ids:
            return []

        placeholders = ", ".join(["?"] * len(user_ids))
        rows = conn.execute(
            f"select user_id, plan_type, signup_source, country, total_revenue, lifetime_events "
            f"from users where user_id in ({placeholders})",
            user_ids,
        ).fetchall()

        return [
            UserProfile(
                user_id=r[0], plan_type=r[1], signup_source=r[2],
                country=r[3], total_revenue=float(r[4] or 0),
                lifetime_events=int(r[5] or 0),
            )
            for r in rows
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()
