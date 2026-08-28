"""
Sequential Probability Ratio Test (SPRT) for early-stopping in A/B experiments.

Unlike fixed-horizon tests that require all data to be collected before analysis,
SPRT allows continuous monitoring with mathematically guaranteed error-rate control.
At each observation the cumulative log-likelihood ratio (LLR) is compared against
two boundaries derived from the desired Type I (alpha) and Type II (beta) error
rates.

Boundaries
----------
Upper (reject H0): B = ln((1 - beta) / alpha)
Lower (accept H0): A = ln(beta / (1 - alpha))

For binary outcomes the per-observation log-likelihood contribution is:

    x_i * ln(theta_1 / theta_0) + (1 - x_i) * ln((1 - theta_1) / (1 - theta_0))

where theta_0 = p_control (null hypothesis) and theta_1 = p_control + delta
(alternative hypothesis with minimum detectable effect *delta*).

Decision rule
-------------
* LLR_n >= B  =>  reject H0 (significant effect detected)
* LLR_n <= A  =>  accept H0 (no meaningful effect)
* A < LLR_n < B  =>  continue testing

References
----------
- Wald, A. *Sequential Analysis* (1947).
- Johari et al. "Always Valid Inference: Continuous Monitoring of A/B Tests"
  (2017, Operations Research).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import duckdb
import numpy as np
from loguru import logger
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class SPRTResult(BaseModel):
    """Snapshot of a sequential test's current state."""

    experiment_id: str
    current_llr: float
    upper_boundary: float  # B (reject H0)
    lower_boundary: float  # A (accept H0)
    decision: str  # "reject_null" | "accept_null" | "continue"
    observations_so_far: int
    estimated_observations_remaining: int | None
    alpha: float
    beta: float
    delta: float
    computed_at: datetime


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------

class SequentialTester:
    """Run the Sequential Probability Ratio Test on experiment data stored in DuckDB.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active connection to the analytics database.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def _fetch_experiment_meta(self, experiment_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "select * from experiments where experiment_id = ?",
            [experiment_id],
        ).fetchone()
        if row is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        cols = [desc[0] for desc in self.conn.description]
        return dict(zip(cols, row))

    def compute_sprt(
        self,
        experiment_id: str,
        alpha: float = 0.05,
        beta: float = 0.20,
        delta: float = 0.02,
    ) -> SPRTResult:
        """Compute SPRT boundaries and the current cumulative LLR.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.
        alpha : float
            Type I error rate (probability of false positive).  Default 0.05.
        beta : float
            Type II error rate (probability of false negative).  Default 0.20.
        delta : float
            Minimum detectable effect on the absolute conversion-rate scale
            (e.g. 0.02 = 2 percentage points).

        Returns
        -------
        SPRTResult
        """
        meta = self._fetch_experiment_meta(experiment_id)
        control_var = meta["control_variant"]
        treatment_var = meta["treatment_variant"]

        # Fetch per-observation outcomes ordered by assignment time
        rows = self.conn.execute(
            """
            select
                ea.variant,
                case when ea.converted then 1 else 0 end as outcome
            from experiment_assignments ea
            where ea.experiment_id = ?
            order by ea.assigned_at
            """,
            [experiment_id],
        ).fetchall()

        # Estimate theta_0 (control rate) from control observations so far
        control_outcomes = [r[1] for r in rows if r[0] == control_var]
        treatment_outcomes = [r[1] for r in rows if r[0] == treatment_var]

        if not control_outcomes:
            raise ValueError("No control observations available for SPRT")

        theta_0 = np.mean(control_outcomes)  # null: true rate = control rate
        theta_1 = theta_0 + delta            # alternative: true rate = control + delta

        # Clamp to (0, 1) to avoid log(0)
        theta_0 = max(min(theta_0, 1 - 1e-10), 1e-10)
        theta_1 = max(min(theta_1, 1 - 1e-10), 1e-10)

        # SPRT boundaries
        upper_boundary = math.log((1 - beta) / alpha)  # B
        lower_boundary = math.log(beta / (1 - alpha))   # A

        # Cumulative log-likelihood ratio over treatment observations
        log_ratio_1 = math.log(theta_1 / theta_0)
        log_ratio_0 = math.log((1 - theta_1) / (1 - theta_0))

        llr = 0.0
        for x_i in treatment_outcomes:
            llr += x_i * log_ratio_1 + (1 - x_i) * log_ratio_0

        # Decision
        if llr >= upper_boundary:
            decision = "reject_null"
        elif llr <= lower_boundary:
            decision = "accept_null"
        else:
            decision = "continue"

        # Rough estimate of remaining observations (Wald's expected sample size)
        expected_info = (
            theta_1 * log_ratio_1 + (1 - theta_1) * log_ratio_0
        )
        if abs(expected_info) > 1e-10 and decision == "continue":
            est_total = int(abs(upper_boundary / expected_info))
            est_remaining = max(est_total - len(treatment_outcomes), 0)
        else:
            est_remaining = None

        n_obs = len(treatment_outcomes)

        logger.info(
            "SPRT for '{}': LLR={:.4f}, boundaries=[{:.4f}, {:.4f}], decision={}, n={}",
            experiment_id,
            llr,
            lower_boundary,
            upper_boundary,
            decision,
            n_obs,
        )

        return SPRTResult(
            experiment_id=experiment_id,
            current_llr=round(llr, 6),
            upper_boundary=round(upper_boundary, 6),
            lower_boundary=round(lower_boundary, 6),
            decision=decision,
            observations_so_far=n_obs,
            estimated_observations_remaining=est_remaining,
            alpha=alpha,
            beta=beta,
            delta=delta,
            computed_at=datetime.now(timezone.utc),
        )
