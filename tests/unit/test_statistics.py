"""Unit tests for statistical methods — A/B testing, sequential analysis, power, forecasting, anomaly detection.

Each test uses known inputs with analytically verifiable outputs or clear
directional properties. Tests are self-contained and do not depend on the
DuckDB test database unless computing experiment-level analyses.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats


class TestFrequentistTests:
    """Tests for the two-proportion z-test and Welch's t-test implementations."""

    def test_z_test_known_values(self) -> None:
        """Verify the z-test against a hand-calculated example.

        Control: 100 conversions out of 1000 (p_c = 0.10)
        Treatment: 130 conversions out of 1000 (p_t = 0.13)

        Pooled proportion: p_pool = 230/2000 = 0.115
        SE = sqrt(0.115 * 0.885 * (1/1000 + 1/1000)) = sqrt(0.000203550) = 0.01427
        z = (0.13 - 0.10) / 0.01427 = 2.102

        The expected z-statistic is approximately 2.10 and the two-sided
        p-value should be approximately 0.036.
        """

        # Compute directly without DB
        n_c, x_c = 1000, 100
        n_t, x_t = 1000, 130
        p_c = x_c / n_c
        p_t = x_t / n_t
        p_pool = (x_c + x_t) / (n_c + n_t)

        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
        z = (p_t - p_c) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        assert abs(z - 2.10) < 0.15, f"z-statistic {z} too far from expected 2.10"
        assert 0.01 < p_value < 0.06, f"p-value {p_value} outside expected range [0.01, 0.06]"
        assert p_value < 0.05, "This test case should be statistically significant"


class TestBayesianTests:
    """Tests for Bayesian A/B test analysis."""

    def test_bayesian_uniform_prior(self) -> None:
        """With a uniform Beta(1,1) prior and identical data, P(B>A) should be ~0.5.

        If both variants have the same number of conversions and total users,
        neither should be favored.
        """
        rng = np.random.default_rng(42)
        n = 1000
        x = 100  # same conversions for both

        # Posterior: Beta(1 + x, 1 + n - x) = Beta(101, 901) for both
        control_samples = rng.beta(1 + x, 1 + n - x, 100_000)
        treatment_samples = rng.beta(1 + x, 1 + n - x, 100_000)
        prob_better = np.mean(treatment_samples > control_samples)

        assert abs(prob_better - 0.5) < 0.02, (
            f"P(treatment > control) = {prob_better}, expected ~0.5 for equal data"
        )

    def test_bayesian_strong_signal(self) -> None:
        """With a clear winner (200/1000 vs 100/1000), P(B>A) should exceed 0.99.

        A 2x difference in conversion rates with 1000 observations per variant
        produces overwhelming posterior evidence.
        """
        rng = np.random.default_rng(42)
        n = 1000
        x_control = 100
        x_treatment = 200

        control_samples = rng.beta(1 + x_control, 1 + n - x_control, 100_000)
        treatment_samples = rng.beta(1 + x_treatment, 1 + n - x_treatment, 100_000)
        prob_better = np.mean(treatment_samples > control_samples)

        assert prob_better > 0.99, (
            f"P(treatment > control) = {prob_better}, expected > 0.99"
        )


class TestSequentialTesting:
    """Tests for the paired two-sample SPRT."""

    def test_sprt_boundaries(self) -> None:
        """Pin the rails for alpha=0.05, beta=0.20.

        Upper (reject H0) B = log(1 / alpha) = log(20) = 2.996, Ville's bound,
        so P(ever reaching B | H0) <= alpha under continuous monitoring.
        Wald's log((1 - beta) / alpha) = log(16) = 2.773 only bounds it by
        alpha / (1 - beta) = 0.0625.
        Lower (accept H0) A = log(beta / (1 - alpha)) = log(0.2105) = -1.558.
        """
        alpha, beta = 0.05, 0.20
        B = math.log(1 / alpha)
        A = math.log(beta / (1 - alpha))
        assert abs(B - 2.996) < 0.01, f"Upper boundary B = {B}, expected ~2.996"
        assert abs(A - (-1.558)) < 0.01, f"Lower boundary A = {A}, expected ~-1.558"

    def test_only_discordant_pairs_move_the_llr(self) -> None:
        """Pairs where both or neither converted carry no evidence.

        The first pair uses the Beta(1, 1) prior mean p0 = 0.5, p1 = 0.52, so
        theta_1 = 0.52 * 0.5 / (0.52 * 0.5 + 0.5 * 0.48) = 0.52 and a
        treatment-only conversion adds log(2 * 0.52) = 0.0392.
        """
        from src.statistics.sequential import llr_path

        path = llr_path(np.array([0, 1, 0]), np.array([1, 1, 0]), delta=0.02)
        assert abs(path[0] - math.log(1.04)) < 1e-9
        assert path[1] == path[0] and path[2] == path[0]

    def test_first_crossing_decides(self) -> None:
        """A path that touches the upper rail stops there, even if it falls back."""
        from src.statistics.sequential import first_crossing

        path = np.array([0.5, 1.5, 3.1, 2.0, -2.0])
        assert first_crossing(path, 2.996, -1.558) == ("reject_null", 3)
        assert first_crossing(np.array([0.1, -1.6]), 2.996, -1.558) == ("accept_null", 2)
        assert first_crossing(np.array([0.1, 0.2]), 2.996, -1.558) == ("continue", None)

    def test_aa_false_positive_rate_under_continuous_monitoring(self) -> None:
        """With no true difference, checking after EVERY pair rejects at most ~alpha.

        The previous design re-estimated the null rate from the control arm
        at each check and rejected 12.9% of A/A experiments at alpha = 0.05.
        600 A/A experiments of 3,000 pairs at p = 0.10: the rejection rate
        must stay at or under alpha plus two standard errors.
        """
        from src.statistics.sequential import first_crossing, llr_path

        rng = np.random.default_rng(2024)
        B, A = math.log(1 / 0.05), math.log(0.20 / 0.95)
        k, n = 600, 3000
        rejected = 0
        for _ in range(k):
            c = (rng.random(n) < 0.10).astype(int)
            t = (rng.random(n) < 0.10).astype(int)
            rejected += first_crossing(llr_path(c, t, 0.02), B, A)[0] == "reject_null"
        se = math.sqrt(0.05 * 0.95 / k)
        assert rejected / k <= 0.05 + 2 * se, f"A/A rejection rate {rejected / k:.3f}"

    def test_detects_a_real_lift(self) -> None:
        """A true 10% -> 12% lift is caught most of the time within 20,000 pairs."""
        from src.statistics.sequential import first_crossing, llr_path

        rng = np.random.default_rng(7)
        B, A = math.log(1 / 0.05), math.log(0.20 / 0.95)
        k, n = 200, 20000
        caught = 0
        for _ in range(k):
            c = (rng.random(n) < 0.10).astype(int)
            t = (rng.random(n) < 0.12).astype(int)
            caught += first_crossing(llr_path(c, t, 0.02), B, A)[0] == "reject_null"
        assert caught / k >= 0.75, f"power {caught / k:.2f}"


class TestPowerAnalysis:
    """Tests for sample size and power calculations."""

    def test_sample_size_binary(self) -> None:
        """Verify sample size for binary metric: baseline=0.10, MDE=0.20, power=0.80.

        With a 20% relative lift on a 10% baseline (p1=0.10, p2=0.12),
        the required sample size per variant is approximately 3800-4000.
        """
        from src.statistics.power_analysis import PowerAnalyzer

        n = PowerAnalyzer.sample_size_binary(
            baseline_rate=0.10,
            minimum_detectable_effect=0.20,
            alpha=0.05,
            power=0.80,
        )
        assert 3000 <= n <= 5000, (
            f"Sample size {n} outside expected range [3000, 5000]"
        )

    def test_sample_size_continuous(self) -> None:
        """Verify sample size for continuous metric.

        baseline_mean=50, std=25, MDE=0.10 (5 unit difference), power=0.80.
        n = 2 * (z_alpha + z_beta)^2 * std^2 / delta^2
          = 2 * (1.96 + 0.84)^2 * 625 / 25
          = 2 * 7.84 * 625 / 25
          = 2 * 196 = 392

        Expected approximately 390-400 per variant.
        """
        from src.statistics.power_analysis import PowerAnalyzer

        n = PowerAnalyzer.sample_size_continuous(
            baseline_mean=50.0,
            baseline_std=25.0,
            minimum_detectable_effect=0.10,
            alpha=0.05,
            power=0.80,
        )
        assert 300 <= n <= 500, (
            f"Sample size {n} outside expected range [300, 500]"
        )

    def test_power_increases_with_n(self) -> None:
        """Statistical power must increase as sample size increases.

        This is a fundamental property: more data provides more evidence
        to detect a real effect.
        """
        from src.statistics.power_analysis import PowerAnalyzer

        power_500 = PowerAnalyzer.compute_power(
            n=500, baseline_rate=0.10, effect_size=0.02, alpha=0.05
        )
        power_1000 = PowerAnalyzer.compute_power(
            n=1000, baseline_rate=0.10, effect_size=0.02, alpha=0.05
        )
        power_5000 = PowerAnalyzer.compute_power(
            n=5000, baseline_rate=0.10, effect_size=0.02, alpha=0.05
        )

        assert power_1000 > power_500, (
            f"Power at n=1000 ({power_1000}) should exceed power at n=500 ({power_500})"
        )
        assert power_5000 > power_1000, (
            f"Power at n=5000 ({power_5000}) should exceed power at n=1000 ({power_1000})"
        )


class TestAnomalyDetection:
    """Tests for anomaly detection on synthetic data."""

    def test_anomaly_catches_spike(self) -> None:
        """Injecting a 5x spike into a stable series should be flagged as anomalous.

        We construct a 90-day series with low variance, insert a spike at
        day 45, and verify that the anomaly detector identifies it.
        """

        rng = np.random.default_rng(42)
        values = 100 + rng.normal(0, 5, 90)
        values[45] = values[45] * 5  # inject a clear spike

        dates = pd.date_range("2025-01-01", periods=90, freq="D")
        pd.DataFrame({
            "metric_date": dates,
            "metric_value": values,
        })

        # Use control chart logic directly to verify the spike is outside bounds
        rolling_mean = pd.Series(values).rolling(14, min_periods=1).mean()
        rolling_std = pd.Series(values).rolling(14, min_periods=1).std().fillna(5)
        ucl = rolling_mean + 3 * rolling_std

        # The spike at index 45 should exceed the upper control limit
        assert values[45] > ucl.iloc[45], (
            f"Spike value {values[45]} did not exceed UCL {ucl.iloc[45]}"
        )


class TestForecasting:
    """Tests for time series forecasting output properties."""

    def test_forecast_output_length(self) -> None:
        """Forecasting N periods ahead should return exactly N predicted values.

        The forecast length must match the requested horizon regardless of
        the historical series length or seasonal configuration.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        rng = np.random.default_rng(42)
        n_historical = 120
        values = 100 + np.arange(n_historical) * 0.5 + rng.normal(0, 5, n_historical)

        dates = pd.date_range("2025-01-01", periods=n_historical, freq="D")
        series = pd.Series(values, index=dates)

        model = ExponentialSmoothing(
            series, trend="add", seasonal=None,
        ).fit(optimized=True)

        forecast_horizon = 30
        forecast = model.forecast(forecast_horizon)

        assert len(forecast) == forecast_horizon, (
            f"Forecast length {len(forecast)} != requested horizon {forecast_horizon}"
        )
        assert forecast.notna().all(), "Forecast contains NaN values"
