"""Unit tests for core analytics computations.

Validates funnel analysis, cohort retention, RFM segmentation, behavioral
clustering, session metrics, and engagement scoring against a small test
dataset loaded into DuckDB.
"""

import duckdb


class TestFunnelAnalysis:
    """Tests for FunnelAnalyzer correctness and boundary conditions."""

    def test_funnel_step_counts_decrease(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """Each funnel step must have fewer or equal users than the previous step.

        This is a fundamental property of ordered funnels: users can only
        drop off, never appear at a later step without completing earlier ones.
        """
        from src.analytics.funnels import FunnelAnalyzer

        analyzer = FunnelAnalyzer(test_db)
        steps = ["view_homepage", "view_product", "click_add_to_cart", "begin_checkout", "complete_purchase"]
        result = analyzer.compute_funnel(steps)

        user_counts = [s.users for s in result.steps]
        for i in range(1, len(user_counts)):
            assert user_counts[i] <= user_counts[i - 1], (
                f"Step {i} ({result.steps[i].step_name}) has more users "
                f"({user_counts[i]}) than step {i-1} ({user_counts[i-1]})"
            )

    def test_funnel_conversion_rates_bounded(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """All conversion rates must be between 0 and 1 inclusive."""
        from src.analytics.funnels import FunnelAnalyzer

        analyzer = FunnelAnalyzer(test_db)
        steps = ["view_homepage", "view_product", "click_add_to_cart"]
        result = analyzer.compute_funnel(steps)

        for step in result.steps:
            assert 0.0 <= step.conversion_rate <= 1.0, (
                f"Step {step.step_name}: conversion_rate {step.conversion_rate} out of [0, 1]"
            )
            assert 0.0 <= step.overall_conversion_rate <= 1.0, (
                f"Step {step.step_name}: overall_conversion_rate "
                f"{step.overall_conversion_rate} out of [0, 1]"
            )


class TestRetentionAnalysis:
    """Tests for RetentionAnalyzer correctness."""

    def test_retention_period_zero_is_100(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """Period 0 retention must be 100% for every cohort.

        By definition, all users in a cohort are active in their signup period.
        """
        from src.analytics.retention import RetentionAnalyzer

        analyzer = RetentionAnalyzer(test_db)
        matrix = analyzer.compute_retention(period="month", num_periods=6)

        for cohort, rates in matrix.retention_rates.items():
            assert len(rates) > 0, f"Cohort {cohort} has no retention rates"
            assert abs(rates[0] - 100.0) < 0.01, (
                f"Cohort {cohort}: period 0 retention is {rates[0]}, expected 100.0"
            )

    def test_retention_rates_decrease(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """Retention rates should generally decrease over time.

        We check that the last period's rate is less than or equal to period 0.
        Small cohorts may show noise, so we only assert the overall trend.
        """
        from src.analytics.retention import RetentionAnalyzer

        analyzer = RetentionAnalyzer(test_db)
        matrix = analyzer.compute_retention(period="month", num_periods=6)

        for cohort, rates in matrix.retention_rates.items():
            if len(rates) > 1:
                assert rates[-1] <= rates[0], (
                    f"Cohort {cohort}: final retention {rates[-1]} > "
                    f"initial retention {rates[0]}"
                )


class TestSegmentation:
    """Tests for RFM and behavioral clustering."""

    def test_rfm_scores_range(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """All RFM scores (R, F, M) must be integers between 1 and 5.

        Scores are computed using quintile binning (ntile(5)), so they
        should always fall in [1, 5].
        """
        from src.analytics.segmentation import SegmentationAnalyzer

        analyzer = SegmentationAnalyzer(test_db)
        rfm = analyzer.compute_rfm()

        for col in ["r_score", "f_score", "m_score"]:
            assert rfm[col].between(1, 5).all(), (
                f"Column {col} has values outside [1, 5]: "
                f"min={rfm[col].min()}, max={rfm[col].max()}"
            )

    def test_rfm_segment_coverage(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """Every user must be assigned to exactly one RFM segment.

        The segmentation logic should be exhaustive -- no user should be
        left without a segment label.
        """
        from src.analytics.segmentation import SegmentationAnalyzer

        analyzer = SegmentationAnalyzer(test_db)
        rfm = analyzer.compute_rfm()

        assert rfm["rfm_segment"].notna().all(), "Some users have null RFM segment"
        assert rfm["user_id"].is_unique, "Duplicate user_ids in RFM output"

    def test_clustering_assigns_all_users(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """K-means clustering must assign every user to a cluster.

        No user should have a null cluster_id after clustering.
        """
        from src.analytics.segmentation import SegmentationAnalyzer

        analyzer = SegmentationAnalyzer(test_db)
        clusters = analyzer.behavioral_clustering(n_clusters=3)

        assert clusters["cluster_id"].notna().all(), "Some users have null cluster_id"
        assert len(clusters) > 0, "Clustering returned empty DataFrame"


class TestSessionMetrics:
    """Tests for session-level metric computations."""

    def test_bounce_rate_bounded(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """Bounce rate must be between 0 and 1 (or 0% and 100%).

        A bounce is a session with exactly one page_view and no other events.
        The rate is the fraction of all sessions that are bounces.
        """
        from src.analytics.sessions import SessionAnalyzer

        analyzer = SessionAnalyzer(test_db)
        metrics = analyzer.compute_session_metrics()

        bounce_rate = metrics.bounce_rate
        assert 0.0 <= bounce_rate <= 1.0, (
            f"Bounce rate {bounce_rate} is outside [0, 1]"
        )

    def test_engagement_score_bounded(self, test_db: duckdb.DuckDBPyConnection) -> None:
        """All engagement scores must fall in [0, 100].

        The score is a min-max normalized composite of session frequency,
        depth, feature breadth, and recency.
        """
        from src.analytics.sessions import SessionAnalyzer

        analyzer = SessionAnalyzer(test_db)
        scores = analyzer.compute_engagement_score()

        assert scores["engagement_score"].between(0, 100).all(), (
            f"Engagement scores outside [0, 100]: "
            f"min={scores['engagement_score'].min()}, "
            f"max={scores['engagement_score'].max()}"
        )
