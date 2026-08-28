"""User segmentation via RFM analysis and behavioral clustering.

Two complementary segmentation approaches:

1. **RFM (Recency-Frequency-Monetary)** -- Quintile scoring on three
   dimensions, mapped to named segments (Champions, Loyal, At Risk,
   Hibernating, New).  Computed entirely in SQL with ``ntile(5)``.

2. **Behavioral clustering** -- K-means on 11 behavioral features
   (session counts, event volume, revenue, feature breadth, timing
   patterns).  Features are z-scored before clustering; silhouette
   score validates the chosen *k*.
"""

from __future__ import annotations

import duckdb
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.analytics.models import ClusterProfile, RFMProfile


def _assign_rfm_segment(r: int, f: int, m: int) -> str:
    """Map RFM quintile scores (1-5) to a named segment.

    Segments follow standard RFM marketing taxonomy:
    - Champions: high on all three dimensions (R>=4, F>=4, M>=4).
    - Loyal: consistently good (R>=3, F>=3, M>=3).
    - At Risk: formerly active users going silent (R<=2, F>=3).
    - Hibernating: low across the board (R<=2, F<=2).
    - New: recently acquired with limited history (R>=4, F<=2).
    """
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3 and m >= 3:
        return "Loyal"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    if r >= 4 and f <= 2:
        return "New"
    return "Other"


class SegmentationAnalyzer:
    """RFM analysis and behavioral clustering for user segmentation.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection to the analytics database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------
    # RFM analysis
    # ------------------------------------------------------------------

    def compute_rfm(self, reference_date: str | None = None) -> pd.DataFrame:
        """Compute RFM scores and segment labels for all users.

        The three raw dimensions:
        - **Recency**: days since the user's most recent event.
        - **Frequency**: count of distinct sessions.
        - **Monetary**: total revenue from purchase events.

        Each dimension is scored 1-5 using ``ntile(5)`` (quintile binning).
        Recency is scored in *descending* order so that lower recency
        (more recent activity) yields a *higher* R score.

        Parameters
        ----------
        reference_date : str, optional
            ISO date used as "today" for recency calculation.  Defaults to
            the most recent event timestamp in the database.

        Returns
        -------
        pd.DataFrame
            Columns: user_id, recency_days, frequency, monetary,
            r_score, f_score, m_score, rfm_segment.
        """
        if reference_date is None:
            row = self.conn.execute(
                "select max(timestamp) from events"
            ).fetchone()
            reference_date = str(row[0])[:10] if row and row[0] else "2025-12-31"

        logger.info("Computing RFM scores (reference_date={})", reference_date)

        sql = f"""
            with rfm_raw as (
                select
                    u.user_id,
                    datediff('day',
                             max(e.timestamp),
                             '{reference_date}'::timestamp) as recency_days,
                    count(distinct e.session_id) as frequency,
                    coalesce(sum(e.revenue), 0) as monetary
                from users u
                left join events e on u.user_id = e.user_id
                group by u.user_id
            )
            select
                user_id,
                recency_days,
                frequency,
                monetary,
                ntile(5) over (order by recency_days desc) as r_score,
                ntile(5) over (order by frequency asc) as f_score,
                ntile(5) over (order by monetary asc) as m_score
            from rfm_raw
        """

        df = self.conn.execute(sql).fetchdf()

        # Map scores to named segments.
        df["rfm_segment"] = df.apply(
            lambda row: _assign_rfm_segment(
                int(row["r_score"]),
                int(row["f_score"]),
                int(row["m_score"]),
            ),
            axis=1,
        )

        segment_counts = df["rfm_segment"].value_counts()
        logger.info("RFM segments: {}", segment_counts.to_dict())
        return df

    def compute_rfm_profiles(self, rfm_df: pd.DataFrame) -> list[RFMProfile]:
        """Aggregate RFM data into per-segment profiles.

        Parameters
        ----------
        rfm_df : pd.DataFrame
            Output of ``compute_rfm``.

        Returns
        -------
        list[RFMProfile]
        """
        total = len(rfm_df)
        profiles: list[RFMProfile] = []
        for seg, group in rfm_df.groupby("rfm_segment"):
            profiles.append(RFMProfile(
                segment=str(seg),
                user_count=len(group),
                avg_recency_days=float(group["recency_days"].mean()),
                avg_frequency=float(group["frequency"].mean()),
                avg_monetary=float(group["monetary"].mean()),
                pct_of_total=round(len(group) / total * 100, 2),
            ))
        return sorted(profiles, key=lambda p: -p.user_count)

    # ------------------------------------------------------------------
    # Behavioral clustering
    # ------------------------------------------------------------------

    def behavioral_clustering(self, n_clusters: int = 5) -> pd.DataFrame:
        """K-means clustering on 11 behavioral features.

        Features extracted per user:
        - total_sessions, total_events, total_page_views
        - avg_session_duration, avg_events_per_session
        - total_revenue, num_purchases
        - days_since_signup, days_since_last_activity
        - unique_features_used (count of distinct ``feature_use`` event names)
        - weekend_ratio (fraction of sessions starting on Sat/Sun)

        All features are z-scored (``StandardScaler``) before clustering.
        Silhouette score is computed to validate cluster quality.

        Parameters
        ----------
        n_clusters : int
            Number of clusters for K-means (default 5).

        Returns
        -------
        pd.DataFrame
            One row per user with all features, cluster_id, and
            cluster_label.
        """
        logger.info("Running behavioral clustering with k={}", n_clusters)

        sql = """
            with user_features as (
                select
                    u.user_id,
                    count(distinct s.session_id) as total_sessions,
                    coalesce(sum(s.event_count), 0) as total_events,
                    coalesce(sum(s.page_view_count), 0) as total_page_views,
                    coalesce(avg(s.duration_seconds), 0) as avg_session_duration,
                    case when count(distinct s.session_id) > 0
                         then coalesce(sum(s.event_count), 0) * 1.0 / count(distinct s.session_id)
                         else 0 end as avg_events_per_session,
                    coalesce(u.total_revenue, 0) as total_revenue,
                    count(distinct case when e.event_name = 'complete_purchase' then e.event_id end) as num_purchases,
                    datediff('day', u.created_at, (select max(timestamp) from events)) as days_since_signup,
                    datediff('day', max(e.timestamp), (select max(timestamp) from events)) as days_since_last_activity,
                    count(distinct case when e.event_type = 'feature_use' then e.event_name end) as unique_features_used,
                    case when count(distinct s.session_id) > 0
                         then sum(case when dayofweek(s.started_at) in (0, 6) then 1 else 0 end) * 1.0
                              / count(distinct s.session_id)
                         else 0 end as weekend_ratio
                from users u
                left join sessions s on u.user_id = s.user_id
                left join events e on u.user_id = e.user_id
                group by u.user_id, u.created_at, u.total_revenue
            )
            select * from user_features
        """

        df = self.conn.execute(sql).fetchdf()

        feature_cols = [
            "total_sessions", "total_events", "total_page_views",
            "avg_session_duration", "avg_events_per_session",
            "total_revenue", "num_purchases",
            "days_since_signup", "days_since_last_activity",
            "unique_features_used", "weekend_ratio",
        ]

        X = df[feature_cols].fillna(0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df["cluster_id"] = labels

        # Silhouette score (skip if only 1 cluster or too few samples).
        sil_score = None
        if n_clusters > 1 and len(df) > n_clusters:
            sil_score = float(silhouette_score(X_scaled, labels, sample_size=min(5000, len(df))))
            logger.info("Silhouette score: {:.3f}", sil_score)

        # Label clusters by dominant characteristic.
        cluster_labels = self._label_clusters(df, feature_cols)
        df["cluster_label"] = df["cluster_id"].map(cluster_labels)

        return df

    def _label_clusters(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[int, str]:
        """Assign descriptive labels to clusters based on centroid characteristics.

        Heuristic: for each cluster, identify the feature most above the
        global mean (in z-score terms) and map to a human-readable label.
        """
        label_map = {
            "total_sessions": "Power Users",
            "total_events": "Power Users",
            "total_page_views": "Browsers",
            "avg_session_duration": "Deep Engagers",
            "avg_events_per_session": "Deep Engagers",
            "total_revenue": "High Spenders",
            "num_purchases": "High Spenders",
            "days_since_signup": "Veterans",
            "days_since_last_activity": "Churning",
            "unique_features_used": "Feature Explorers",
            "weekend_ratio": "Weekend Warriors",
        }

        fallback_labels = [
            "Power Users", "Window Shoppers", "Weekend Warriors",
            "Churning", "New Explorers",
        ]

        global_means = df[feature_cols].mean()
        global_stds = df[feature_cols].std().replace(0, 1)

        used_labels: set[str] = set()
        cluster_labels: dict[int, str] = {}

        for cid in sorted(df["cluster_id"].unique()):
            cluster_data = df[df["cluster_id"] == cid][feature_cols].mean()
            z_scores = (cluster_data - global_means) / global_stds

            # Pick the feature with highest z-score that maps to an unused label.
            for feat in z_scores.sort_values(ascending=False).index:
                candidate = label_map.get(feat, "Other")
                if candidate not in used_labels:
                    cluster_labels[cid] = candidate
                    used_labels.add(candidate)
                    break
            else:
                # All preferred labels taken; use a fallback.
                for fb in fallback_labels:
                    if fb not in used_labels:
                        cluster_labels[cid] = fb
                        used_labels.add(fb)
                        break
                else:
                    cluster_labels[cid] = f"Cluster {cid}"

        return cluster_labels

    def compute_cluster_profiles(
        self,
        clustered_df: pd.DataFrame,
    ) -> list[ClusterProfile]:
        """Aggregate clustered user data into per-cluster profiles.

        Parameters
        ----------
        clustered_df : pd.DataFrame
            Output of ``behavioral_clustering``.

        Returns
        -------
        list[ClusterProfile]
        """
        profiles: list[ClusterProfile] = []
        for cid, group in clustered_df.groupby("cluster_id"):
            profiles.append(ClusterProfile(
                cluster_id=int(cid),
                cluster_label=group["cluster_label"].iloc[0],
                user_count=len(group),
                avg_total_sessions=float(group["total_sessions"].mean()),
                avg_total_events=float(group["total_events"].mean()),
                avg_total_revenue=float(group["total_revenue"].mean()),
                avg_session_duration=float(group["avg_session_duration"].mean()),
                avg_unique_features=float(group["unique_features_used"].mean()),
            ))
        return sorted(profiles, key=lambda p: -p.user_count)
