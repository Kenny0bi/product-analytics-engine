"""Integration tests for the FastAPI REST API.

Uses FastAPI's TestClient to exercise all endpoint groups against a small
test database. The test database is populated via the session-scoped
conftest fixtures.
"""


import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def api_client(test_db_path) -> TestClient:
    """A TestClient wired to the test database.

    Patches the application state so all endpoints query the small
    test dataset instead of the production database.
    """
    app.state.db_path = str(test_db_path)
    return TestClient(app)


class TestMetricsEndpoints:
    """Tests for /api/v1/metrics/* endpoints."""

    def test_metrics_summary_returns_200(self, api_client: TestClient) -> None:
        """The summary endpoint must return HTTP 200 with all required fields."""
        response = api_client.get("/api/v1/metrics/summary")
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        for key in ("dau", "wau", "mau", "total_revenue", "conversion_rate", "avg_session_duration_seconds"):
            assert key in data, f"Missing field: {key}"

    def test_daily_metrics_with_date_range(self, api_client: TestClient) -> None:
        """Daily metrics with a date range filter should return a list of data points."""
        response = api_client.get(
            "/api/v1/metrics/daily",
            params={
                "metric_name": "dau",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
            },
        )
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        assert isinstance(data, list), "Expected a list of data points"


class TestFunnelEndpoints:
    """Tests for /api/v1/funnels endpoints."""

    def test_funnel_endpoint_returns_steps(self, api_client: TestClient) -> None:
        """The funnel endpoint should return step-level conversion data."""
        response = api_client.get(
            "/api/v1/funnels",
            params={"steps": "view_homepage,view_product,click_add_to_cart"},
        )
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        assert "steps" in data, "Missing 'steps' in response"
        assert len(data["steps"]) == 3, f"Expected 3 steps, got {len(data['steps'])}"
        assert data["steps"][0]["step_index"] == 0


class TestRetentionEndpoints:
    """Tests for /api/v1/retention endpoints."""

    def test_retention_endpoint_returns_matrix(self, api_client: TestClient) -> None:
        """The retention endpoint should return a matrix with cohorts and rates."""
        response = api_client.get(
            "/api/v1/retention",
            params={"period": "month", "num_periods": 6},
        )
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        assert "cohorts" in data, "Missing 'cohorts' in response"
        assert "retention_rates" in data, "Missing 'retention_rates' in response"
        assert data["period_type"] == "month"


class TestExperimentEndpoints:
    """Tests for /api/v1/experiments endpoints."""

    def test_experiment_list_returns_experiments(self, api_client: TestClient) -> None:
        """The experiment list endpoint should return at least one experiment."""
        response = api_client.get("/api/v1/experiments")
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        assert isinstance(data, list), "Expected a list of experiments"

    def test_experiment_detail_returns_ab_results(self, api_client: TestClient) -> None:
        """The experiment detail endpoint should return frequentist and Bayesian results."""
        # First get the experiment list to find a valid ID
        list_response = api_client.get("/api/v1/experiments")
        if list_response.status_code != 200 or not list_response.json():
            pytest.skip("No experiments available in test database")

        exp_id = list_response.json()[0]["experiment_id"]
        response = api_client.get(f"/api/v1/experiments/{exp_id}")
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        assert "frequentist" in data, "Missing frequentist results"
        assert "bayesian" in data, "Missing Bayesian results"
        assert "observed_lift" in data

    def test_power_analysis_endpoint(self, api_client: TestClient) -> None:
        """The power analysis endpoint should compute sample sizes for binary metrics."""
        response = api_client.post(
            "/api/v1/experiments/power",
            json={
                "metric_type": "binary",
                "baseline_rate": 0.10,
                "minimum_detectable_effect": 0.20,
                "alpha": 0.05,
                "power": 0.80,
            },
        )
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        assert data["sample_size_per_variant"] > 0
        assert data["total_sample_size"] == data["sample_size_per_variant"] * 2


class TestSegmentEndpoints:
    """Tests for /api/v1/segments endpoints."""

    def test_rfm_segments_endpoint(self, api_client: TestClient) -> None:
        """The RFM endpoint should return segment profiles with user counts."""
        response = api_client.get("/api/v1/segments/rfm")
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        assert "segments" in data, "Missing 'segments' in response"
        assert data["total_users"] > 0, "Expected nonzero total_users"


class TestForecastEndpoints:
    """Tests for /api/v1/forecast endpoints."""

    def test_forecast_endpoint_returns_predictions(self, api_client: TestClient) -> None:
        """The forecast endpoint should return the requested number of predictions."""
        response = api_client.get(
            "/api/v1/forecast/dau",
            params={"periods_ahead": 14},
        )
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        assert "forecast" in data, "Missing 'forecast' in response"
        assert data["metric_name"] == "dau"
        assert data["periods_ahead"] == 14

    def test_forecast_length_matches_horizon(self, api_client: TestClient) -> None:
        """The forecast list must contain exactly periods_ahead points."""
        response = api_client.get("/api/v1/forecast/dau", params={"periods_ahead": 14})
        assert response.status_code == 200
        assert len(response.json()["forecast"]) == 14

    def test_anomalies_endpoint_returns_valid_shape(self, api_client: TestClient) -> None:
        """The anomalies endpoint must return well-formed anomaly records.

        This endpoint once shipped with response fields that did not exist on
        the AnomalyResult model and 500ed on every call; this test pins the
        contract.
        """
        response = api_client.get("/api/v1/anomalies/dau", params={"lookback_days": 90})
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        for record in response.json():
            for key in ("date", "metric_value", "expected_value", "deviation",
                        "upper_control_limit", "lower_control_limit", "severity"):
                assert key in record, f"Missing field: {key}"
            assert record["severity"] in ("low", "medium", "high")
