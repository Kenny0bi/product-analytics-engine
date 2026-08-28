"""
Sample-size and statistical-power calculations for experiment planning.

All formulas assume independent two-sample comparisons (control vs. treatment)
with equal allocation (1:1 split).

Binary metrics
--------------
Uses the unpooled two-proportion formula:

    n = ((z_{alpha} * sqrt(2 * p_bar * (1 - p_bar))
          + z_{beta} * sqrt(p1*(1-p1) + p2*(1-p2))) / (p2 - p1))^2

where p_bar = (p1 + p2) / 2.

Continuous metrics
------------------
Standard two-sample t-test formula assuming equal variances:

    n = 2 * (z_{alpha} + z_{beta})^2 * sigma^2 / delta^2

Power computation
-----------------
Given a fixed sample size, compute the probability of rejecting H0 under
the alternative hypothesis using the non-central normal approximation.

References
----------
- Lehr's rule of thumb: n ~ 16 / delta^2 for 80 % power.
- Lachin, J. *Biostatistical Methods*, 2nd ed., Ch. 2-3.
- scipy.stats.norm for quantile functions.
"""

from __future__ import annotations

import math

from loguru import logger
from scipy.stats import norm


class PowerAnalyzer:
    """Static methods for sample-size estimation and power computation."""

    @staticmethod
    def sample_size_binary(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        two_sided: bool = True,
    ) -> int:
        """Required sample size per variant for a binary (proportion) metric.

        Parameters
        ----------
        baseline_rate : float
            Expected conversion rate under control (e.g. 0.10 for 10 %).
        minimum_detectable_effect : float
            Relative lift to detect (e.g. 0.20 means a 20 % relative
            increase from baseline).
        alpha : float
            Significance level (default 0.05).
        power : float
            Desired statistical power (default 0.80).
        two_sided : bool
            Whether the test is two-sided (default True).

        Returns
        -------
        int
            Minimum number of observations per variant (ceiling).
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        p_bar = (p1 + p2) / 2.0

        z_alpha = norm.ppf(1 - alpha / (2 if two_sided else 1))
        z_beta = norm.ppf(power)

        numerator = (
            z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        )
        denominator = abs(p2 - p1)

        if denominator == 0:
            raise ValueError("MDE of zero produces infinite sample size")

        n = (numerator / denominator) ** 2
        n = math.ceil(n)

        logger.info(
            "Binary sample size: baseline={:.3f}, mde={:.2%}, alpha={}, power={} => n={}",
            baseline_rate,
            minimum_detectable_effect,
            alpha,
            power,
            n,
        )
        return n

    @staticmethod
    def sample_size_continuous(
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """Required sample size per variant for a continuous metric.

        Parameters
        ----------
        baseline_mean : float
            Expected mean under control.
        baseline_std : float
            Expected standard deviation (assumed equal across variants).
        minimum_detectable_effect : float
            Relative lift to detect (e.g. 0.10 for a 10 % increase in mean).
        alpha : float
        power : float

        Returns
        -------
        int
        """
        delta = baseline_mean * minimum_detectable_effect
        if delta == 0:
            raise ValueError("MDE of zero produces infinite sample size")

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) ** 2) * (baseline_std ** 2) / (delta ** 2)
        n = math.ceil(n)

        logger.info(
            "Continuous sample size: mean={:.2f}, std={:.2f}, mde={:.2%} => n={}",
            baseline_mean,
            baseline_std,
            minimum_detectable_effect,
            n,
        )
        return n

    @staticmethod
    def compute_power(
        n: int,
        baseline_rate: float,
        effect_size: float,
        alpha: float = 0.05,
    ) -> float:
        """Compute statistical power given a fixed sample size.

        Uses the normal approximation for a two-proportion z-test.

        Parameters
        ----------
        n : int
            Sample size per variant.
        baseline_rate : float
            Control conversion rate.
        effect_size : float
            Relative lift (e.g. 0.20).
        alpha : float

        Returns
        -------
        float
            Power (probability of correctly rejecting H0).
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + effect_size)
        p_bar = (p1 + p2) / 2.0

        z_alpha = norm.ppf(1 - alpha / 2)

        se_null = math.sqrt(2 * p_bar * (1 - p_bar) / n)
        se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)

        if se_alt == 0:
            return 1.0

        z_beta = (abs(p2 - p1) - z_alpha * se_null) / se_alt
        power = float(norm.cdf(z_beta))

        logger.debug(
            "Power computation: n={}, baseline={:.3f}, effect={:.2%} => power={:.4f}",
            n,
            baseline_rate,
            effect_size,
            power,
        )
        return round(power, 4)
