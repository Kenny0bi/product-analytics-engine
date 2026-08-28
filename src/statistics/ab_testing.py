"""
A/B test analysis with frequentist and Bayesian methods.

Frequentist: two-proportion z-test for binary outcomes, Welch's t-test for
continuous outcomes, chi-squared test for independence.

Bayesian: Beta-Bernoulli conjugate model for binary outcomes with Monte Carlo
posterior sampling to compute P(treatment > control), expected relative lift,
credible intervals, and expected loss for each decision.

References
----------
- Kohavi, Tang, Xu. *Trustworthy Online Controlled Experiments* (2020), Ch. 17-19.
- Gelman et al. *Bayesian Data Analysis*, 3rd ed., Ch. 5 (hierarchical priors for A/B).
- Cohen, J. *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import duckdb
import numpy as np
from loguru import logger
from pydantic import BaseModel
from scipy import stats

# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------

class FrequentistResult(BaseModel):
    """Result of a frequentist hypothesis test (z-test, t-test, or chi-squared)."""

    test_type: str  # "z-test", "welch-t", "chi-squared"
    test_statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: float  # Cohen's h (binary) or Cohen's d (continuous)
    statistical_power: float
    is_significant: bool  # p < 0.05


class BayesianResult(BaseModel):
    """Result of a Bayesian A/B analysis via posterior sampling."""

    prob_treatment_better: float
    expected_lift: float
    lift_credible_interval: tuple[float, float]  # 95 % HDI
    expected_loss_treatment: float
    expected_loss_control: float
    recommendation: str  # "Choose treatment" | "Choose control" | "Continue testing"


class ExperimentResult(BaseModel):
    """Combined frequentist + Bayesian analysis for a single experiment."""

    experiment_id: str
    experiment_name: str
    metric_name: str
    control_metric: float
    treatment_metric: float
    observed_lift: float
    frequentist: FrequentistResult
    bayesian: BayesianResult
    sample_sizes: dict[str, int]
    computed_at: datetime


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ABTestAnalyzer:
    """Run frequentist and Bayesian A/B test analysis against DuckDB experiment data.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active connection to the analytics DuckDB database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_experiment_meta(self, experiment_id: str) -> dict[str, Any]:
        """Return the experiment row as a dict, raising if not found."""
        row = self.conn.execute(
            "select * from experiments where experiment_id = ?",
            [experiment_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        cols = [desc[0] for desc in self.conn.description]
        return dict(zip(cols, row))

    def _fetch_variant_stats(
        self, experiment_id: str
    ) -> dict[str, dict[str, Any]]:
        """Return per-variant aggregates: n, conversions, mean value, std value."""
        rows = self.conn.execute(
            """
            select
                ea.variant,
                count(*)                             as n,
                sum(case when ea.converted then 1 else 0 end) as conversions,
                avg(ea.conversion_value)             as mean_value,
                stddev_samp(ea.conversion_value)     as std_value
            from experiment_assignments ea
            where ea.experiment_id = ?
            group by ea.variant
            """,
            [experiment_id],
        ).fetchall()
        cols = [desc[0] for desc in self.conn.description]
        return {r[0]: dict(zip(cols, r)) for r in rows}

    # ------------------------------------------------------------------
    # Frequentist
    # ------------------------------------------------------------------

    def frequentist_test(
        self,
        experiment_id: str,
        metric_type: str = "binary",
    ) -> FrequentistResult:
        """Run a frequentist significance test.

        For *binary* metrics (e.g. conversion rate) the two-proportion z-test is
        used:

            z = (p_t - p_c) / sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))

        where p_pool = (x_c + x_t) / (n_c + n_t).  A chi-squared test is also
        computed internally; only the z-test result is returned because the two
        are algebraically equivalent for 2x2 tables (chi2 = z^2).

        For *continuous* metrics (e.g. revenue per user) Welch's t-test is used:

            t = (mean_t - mean_c) / sqrt(s_c^2/n_c + s_t^2/n_t)

        with Welch-Satterthwaite degrees of freedom.

        Parameters
        ----------
        experiment_id : str
            Identifier of the experiment in the ``experiments`` table.
        metric_type : str
            ``"binary"`` for conversion-style metrics, ``"continuous"`` for
            revenue / duration metrics.

        Returns
        -------
        FrequentistResult
        """
        meta = self._fetch_experiment_meta(experiment_id)
        vs = self._fetch_variant_stats(experiment_id)
        control_var = meta["control_variant"]
        treatment_var = meta["treatment_variant"]
        c = vs[control_var]
        t = vs[treatment_var]

        n_c, n_t = int(c["n"]), int(t["n"])

        if metric_type == "binary":
            x_c = int(c["conversions"])
            x_t = int(t["conversions"])
            p_c = x_c / n_c
            p_t = x_t / n_t
            p_pool = (x_c + x_t) / (n_c + n_t)
            se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            z = (p_t - p_c) / se if se > 0 else 0.0
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            # 95 % confidence interval on the difference p_t - p_c
            se_diff = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
            ci = (
                (p_t - p_c) - 1.96 * se_diff,
                (p_t - p_c) + 1.96 * se_diff,
            )

            # Cohen's h = 2 * arcsin(sqrt(p_t)) - 2 * arcsin(sqrt(p_c))
            effect_size = 2 * math.asin(math.sqrt(p_t)) - 2 * math.asin(math.sqrt(p_c))

            # Chi-squared (for logging / cross-check)
            observed = np.array([[x_c, n_c - x_c], [x_t, n_t - x_t]])
            chi2, chi2_p, _, _ = stats.chi2_contingency(observed, correction=False)
            logger.debug(
                "Chi-squared cross-check: chi2={:.4f}, p={:.4f}", chi2, chi2_p
            )

            # Power: P(reject H0 | true effect = observed)
            ncp = abs(p_t - p_c) / se if se > 0 else 0.0
            power = 1 - stats.norm.cdf(1.96 - ncp) + stats.norm.cdf(-1.96 - ncp)

            test_type = "z-test"
            test_stat = z

        else:
            # Continuous: Welch's t-test
            mean_c = float(c["mean_value"] or 0)
            mean_t = float(t["mean_value"] or 0)
            std_c = float(c["std_value"] or 1)
            std_t = float(t["std_value"] or 1)

            se = math.sqrt(std_c**2 / n_c + std_t**2 / n_t)
            t_stat = (mean_t - mean_c) / se if se > 0 else 0.0

            # Welch-Satterthwaite degrees of freedom
            num = (std_c**2 / n_c + std_t**2 / n_t) ** 2
            denom = (std_c**2 / n_c) ** 2 / (n_c - 1) + (std_t**2 / n_t) ** 2 / (n_t - 1)
            df = num / denom if denom > 0 else 1.0

            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            ci = (
                (mean_t - mean_c) - stats.t.ppf(0.975, df) * se,
                (mean_t - mean_c) + stats.t.ppf(0.975, df) * se,
            )

            # Cohen's d
            pooled_std = math.sqrt(
                ((n_c - 1) * std_c**2 + (n_t - 1) * std_t**2) / (n_c + n_t - 2)
            )
            effect_size = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0.0

            # Power via non-central t
            ncp = abs(mean_t - mean_c) / se if se > 0 else 0.0
            critical = stats.t.ppf(0.975, df)
            power = 1 - stats.nct.cdf(critical, df, ncp) + stats.nct.cdf(-critical, df, ncp)

            test_type = "welch-t"
            test_stat = t_stat

        logger.info(
            "Frequentist test for '{}': {} stat={:.4f}, p={:.4f}, significant={}",
            experiment_id,
            test_type,
            test_stat,
            p_value,
            p_value < 0.05,
        )

        return FrequentistResult(
            test_type=test_type,
            test_statistic=round(test_stat, 6),
            p_value=round(p_value, 6),
            confidence_interval=(round(ci[0], 6), round(ci[1], 6)),
            effect_size=round(effect_size, 6),
            statistical_power=round(power, 4),
            is_significant=p_value < 0.05,
        )

    # ------------------------------------------------------------------
    # Bayesian
    # ------------------------------------------------------------------

    def bayesian_test(
        self,
        experiment_id: str,
        metric_type: str = "binary",
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        num_samples: int = 100_000,
    ) -> BayesianResult:
        """Run a Bayesian A/B analysis via posterior sampling.

        For binary metrics the conjugate Beta-Bernoulli model is used:

        * Prior:      Beta(alpha, beta) for both variants (default uniform).
        * Posterior:   Beta(alpha + successes, beta + failures).
        * Inference:   Draw ``num_samples`` from each posterior and compute:
          - P(treatment > control) = mean(samples_t > samples_c)
          - Expected relative lift = E[(samples_t - samples_c) / samples_c]
          - 95 % credible interval on relative lift
          - Expected loss = E[max(samples_c - samples_t, 0)]  (risk of
            choosing treatment when control is actually better)

        For continuous metrics a Normal model with empirical variance is used
        as a pragmatic approximation (Normal-InverseGamma conjugate with a
        weakly informative prior converges to the data likelihood quickly).

        Parameters
        ----------
        experiment_id : str
        metric_type : str
        prior_alpha, prior_beta : float
            Shape parameters of the Beta prior (binary) or scale of the
            inverse-gamma prior (continuous).
        num_samples : int
            Number of Monte Carlo posterior draws.

        Returns
        -------
        BayesianResult
        """
        meta = self._fetch_experiment_meta(experiment_id)
        vs = self._fetch_variant_stats(experiment_id)
        control_var = meta["control_variant"]
        treatment_var = meta["treatment_variant"]
        c = vs[control_var]
        t = vs[treatment_var]

        rng = np.random.default_rng(42)

        if metric_type == "binary":
            x_c, n_c = int(c["conversions"]), int(c["n"])
            x_t, n_t = int(t["conversions"]), int(t["n"])

            # Posterior samples from Beta(alpha + x, beta + n - x)
            samples_c = rng.beta(
                prior_alpha + x_c, prior_beta + n_c - x_c, size=num_samples
            )
            samples_t = rng.beta(
                prior_alpha + x_t, prior_beta + n_t - x_t, size=num_samples
            )
        else:
            # Continuous: approximate Normal posterior
            mean_c = float(c["mean_value"] or 0)
            mean_t = float(t["mean_value"] or 0)
            std_c = float(c["std_value"] or 1)
            std_t = float(t["std_value"] or 1)
            n_c, n_t = int(c["n"]), int(t["n"])

            samples_c = rng.normal(mean_c, std_c / math.sqrt(n_c), size=num_samples)
            samples_t = rng.normal(mean_t, std_t / math.sqrt(n_t), size=num_samples)

        prob_treatment_better = float(np.mean(samples_t > samples_c))

        # Relative lift distribution, guarding against division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            lift_samples = np.where(
                samples_c > 0,
                (samples_t - samples_c) / samples_c,
                0.0,
            )
        expected_lift = float(np.mean(lift_samples))
        lift_ci = (
            float(np.percentile(lift_samples, 2.5)),
            float(np.percentile(lift_samples, 97.5)),
        )

        # Expected loss for each decision
        loss_treatment = float(np.mean(np.maximum(samples_c - samples_t, 0)))
        loss_control = float(np.mean(np.maximum(samples_t - samples_c, 0)))

        # Decision heuristic
        if prob_treatment_better > 0.95:
            recommendation = "Choose treatment"
        elif prob_treatment_better < 0.05:
            recommendation = "Choose control"
        else:
            recommendation = "Continue testing"

        logger.info(
            "Bayesian test for '{}': P(treatment better)={:.3f}, E[lift]={:.3f}",
            experiment_id,
            prob_treatment_better,
            expected_lift,
        )

        return BayesianResult(
            prob_treatment_better=round(prob_treatment_better, 4),
            expected_lift=round(expected_lift, 4),
            lift_credible_interval=(round(lift_ci[0], 4), round(lift_ci[1], 4)),
            expected_loss_treatment=round(loss_treatment, 6),
            expected_loss_control=round(loss_control, 6),
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Combined analysis
    # ------------------------------------------------------------------

    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Run both frequentist and Bayesian analysis for an experiment.

        Automatically infers the metric type from the experiment definition:
        ``conversion_rate`` and retention metrics are binary; revenue and
        duration metrics are continuous.

        Parameters
        ----------
        experiment_id : str

        Returns
        -------
        ExperimentResult
        """
        meta = self._fetch_experiment_meta(experiment_id)
        metric_name = meta["metric_name"]

        binary_metrics = {"conversion_rate", "7_day_retention", "retention"}
        metric_type = "binary" if metric_name in binary_metrics else "continuous"

        freq = self.frequentist_test(experiment_id, metric_type=metric_type)
        bayes = self.bayesian_test(experiment_id, metric_type=metric_type)

        vs = self._fetch_variant_stats(experiment_id)
        control_var = meta["control_variant"]
        treatment_var = meta["treatment_variant"]
        c = vs[control_var]
        t = vs[treatment_var]

        if metric_type == "binary":
            control_metric = int(c["conversions"]) / int(c["n"])
            treatment_metric = int(t["conversions"]) / int(t["n"])
        else:
            control_metric = float(c["mean_value"] or 0)
            treatment_metric = float(t["mean_value"] or 0)

        observed_lift = (
            (treatment_metric - control_metric) / control_metric
            if control_metric > 0
            else 0.0
        )

        logger.info(
            "Experiment '{}' ({}): control={:.4f}, treatment={:.4f}, lift={:.2%}",
            experiment_id,
            metric_name,
            control_metric,
            treatment_metric,
            observed_lift,
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=meta["experiment_name"],
            metric_name=metric_name,
            control_metric=round(control_metric, 6),
            treatment_metric=round(treatment_metric, 6),
            observed_lift=round(observed_lift, 4),
            frequentist=freq,
            bayesian=bayes,
            sample_sizes={
                control_var: int(c["n"]),
                treatment_var: int(t["n"]),
            },
            computed_at=datetime.now(timezone.utc),
        )
