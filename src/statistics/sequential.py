"""
Sequential Probability Ratio Test (SPRT) for early-stopping in A/B experiments.

Unlike fixed-horizon tests that require all data to be collected before analysis,
a sequential test can be checked after every user and still keep its false
positive rate at alpha. That guarantee only holds if the test is built so the
null hypothesis is fixed before the data arrives.

Why the test is run on pairs
----------------------------
The first version of this module set the null rate theta_0 to the control arm's
observed conversion rate and re-estimated it at every check. Under that design
H0 moves with the data, the likelihood ratio is no longer a martingale under the
null, and Wald's error bounds stop applying. Simulated A/A experiments (no true
difference, 10 checks) rejected H0 12.9% of the time at alpha = 0.05.

This version uses the classic two-sample construction. Control and treatment
users are paired in assignment order. A pair where both or neither converted
says nothing about which arm is better. In a pair where exactly one converted
(a discordant pair), H0 "both arms share a rate" means the treatment user is
the converter with probability exactly 1/2, whatever that shared rate is. So
the null is exact and needs no estimate.

The alternative is "treatment is delta above control". For a discordant pair it
says the treatment user converts with probability

    theta_1 = p_1 (1 - p_0) / (p_1 (1 - p_0) + p_0 (1 - p_1)),  p_1 = p_0 + delta

p_0 is unknown, so it is estimated from the control users in EARLIER pairs only
(with a Beta(1, 1) prior). Because each pair's alternative is fixed before that
pair is seen, the likelihood ratio stays a non-negative martingale with mean 1
under H0, and Ville's inequality bounds P(it ever reaches 1/alpha) by alpha, no
matter how often it is checked.

Boundaries
----------
Upper (reject H0): B = ln(1 / alpha)          (Ville; Wald's ln((1-beta)/alpha)
                                               only bounds the error by alpha/(1-beta))
Lower (accept H0): A = ln(beta / (1 - alpha)) (Wald)

Decision rule
-------------
The test stops at the FIRST pair where the cumulative LLR leaves (A, B):
* crosses B first  =>  reject H0 (treatment better by about delta)
* crosses A first  =>  accept H0 (no effect of size delta)
* never crossed    =>  continue

References
----------
- Wald, A. *Sequential Analysis* (1947), comparison of two binomial proportions.
- Ville, J. *Etude critique de la notion de collectif* (1939).
- Johari et al. "Always Valid Inference: Continuous Monitoring of A/B Tests"
  (Operations Research, 2022).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import duckdb
import numpy as np
from loguru import logger
from pydantic import BaseModel


def _theta_1(p0: float | np.ndarray, p1: float | np.ndarray) -> Any:
    """P(treatment user is the converter | discordant pair) under H1."""
    return p1 * (1 - p0) / (p1 * (1 - p0) + p0 * (1 - p1))


def llr_path(control: np.ndarray, treatment: np.ndarray, delta: float) -> np.ndarray:
    """Cumulative log-likelihood ratio after each control/treatment pair.

    Pair k contributes ln(2 theta_1) if only the treatment user converted,
    ln(2 (1 - theta_1)) if only the control user converted, and 0 otherwise.
    theta_1 for pair k uses the control rate from pairs 0..k-1 only, so the
    alternative is always fixed before the pair it scores.
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    k = np.arange(len(c))
    seen = np.concatenate([[0.0], np.cumsum(c)[:-1]])
    p0 = (seen + 1) / (k + 2)
    p1 = np.minimum(p0 + delta, 1 - 1e-9)
    th = _theta_1(p0, p1)
    inc = np.where(t > c, np.log(2 * th), np.where(t < c, np.log(2 * (1 - th)), 0.0))
    return np.cumsum(inc)


def first_crossing(path: np.ndarray, upper: float, lower: float) -> tuple[str, int | None]:
    """Decision at the first pair where the path leaves (lower, upper).

    Returns the decision and the 1-based pair number where it was made.
    """
    up = np.flatnonzero(path >= upper)
    lo = np.flatnonzero(path <= lower)
    if up.size == 0 and lo.size == 0:
        return "continue", None
    if lo.size == 0 or (up.size and up[0] < lo[0]):
        return "reject_null", int(up[0]) + 1
    return "accept_null", int(lo[0]) + 1


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
    observations_so_far: int  # user pairs (one control + one treatment)
    discordant_pairs: int  # pairs where exactly one user converted
    stopped_at: int | None  # pair at which the LLR first left the corridor
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
        """Run the paired two-sample SPRT on everything observed so far.

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

        control_outcomes = np.array([r[1] for r in rows if r[0] == control_var])
        treatment_outcomes = np.array([r[1] for r in rows if r[0] == treatment_var])
        n_obs = min(len(control_outcomes), len(treatment_outcomes))
        if n_obs == 0:
            raise ValueError("SPRT needs at least one control and one treatment observation")

        upper_boundary = math.log(1 / alpha)          # B (Ville)
        lower_boundary = math.log(beta / (1 - alpha))  # A (Wald)

        path = llr_path(control_outcomes[:n_obs], treatment_outcomes[:n_obs], delta)
        llr = float(path[-1])
        decision, stop = first_crossing(path, upper_boundary, lower_boundary)
        discordant = int(np.sum(control_outcomes[:n_obs] != treatment_outcomes[:n_obs]))

        # Wald's expected sample size under H1, in pairs: B divided by the
        # expected LLR gain per pair (P(discordant) * KL(theta_1 || 1/2)).
        p0 = (control_outcomes[:n_obs].sum() + 1) / (n_obs + 2)
        p1 = min(p0 + delta, 1 - 1e-9)
        theta_1 = _theta_1(p0, p1)
        p_disc = p1 * (1 - p0) + p0 * (1 - p1)
        gain = p_disc * (theta_1 * math.log(2 * theta_1)
                         + (1 - theta_1) * math.log(2 * (1 - theta_1)))
        est_remaining: int | None = None
        if decision == "continue" and gain > 1e-12:
            est_remaining = max(int(math.ceil((upper_boundary - llr) / gain)), 0)

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
            discordant_pairs=discordant,
            stopped_at=stop,
            estimated_observations_remaining=est_remaining,
            alpha=alpha,
            beta=beta,
            delta=delta,
            computed_at=datetime.now(timezone.utc),
        )
