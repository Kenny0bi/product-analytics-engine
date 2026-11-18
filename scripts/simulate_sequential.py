"""Reproduce every sequential-testing number quoted in the README.

Part 1, the bug. 1,000 A/A experiments (both arms convert at 10%), 2,000 users
per arm arriving in 10 looks of 200, written into an in-memory DuckDB with the
real schema and read back through this repo's own code:

* fixed horizon: ABTestAnalyzer's z-test, read once at the end
* peeking: the same z-test read at every look, stop at the first p < 0.05
* old SPRT: the first version of SequentialTester (null rate re-estimated
  from the control arm at every look), reproduced in old_sprt_decision()
* paired SPRT: the current SequentialTester, checked at the same 10 looks

Part 2, the fix under the harshest schedule: the paired SPRT checked after
EVERY pair, 4,000 A/A experiments of 20,000 pairs at base rates of 3%, 10% and
30%, then power against a true 10% -> 12% lift.

Takes several minutes. Usage: python scripts/simulate_sequential.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistics.ab_testing import ABTestAnalyzer  # noqa: E402
from src.statistics.sequential import (  # noqa: E402
    SequentialTester,
    first_crossing,
    llr_path,
)

ALPHA, BETA, DELTA = 0.05, 0.20, 0.02
WALD_UPPER = math.log((1 - BETA) / ALPHA)
LOWER = math.log(BETA / (1 - ALPHA))
UPPER = math.log(1 / ALPHA)


def old_sprt_decision(control: np.ndarray, treatment: np.ndarray) -> str:
    """The first SequentialTester: theta_0 re-estimated from control every check."""
    theta_0 = min(max(float(np.mean(control)), 1e-10), 1 - 1e-10)
    theta_1 = min(max(theta_0 + DELTA, 1e-10), 1 - 1e-10)
    r1, r0 = math.log(theta_1 / theta_0), math.log((1 - theta_1) / (1 - theta_0))
    llr = float(np.sum(treatment * r1 + (1 - treatment) * r0))
    if llr >= WALD_UPPER:
        return "reject_null"
    if llr <= LOWER:
        return "accept_null"
    return "continue"


def part_one(k: int = 1000, looks: int = 10, per_look: int = 200, p: float = 0.10) -> None:
    rng = np.random.default_rng(7)
    con = duckdb.connect()
    con.execute("create table experiments (experiment_id varchar, "
                "control_variant varchar, treatment_variant varchar)")
    con.execute("create table experiment_assignments (experiment_id varchar, "
                "variant varchar, converted boolean, conversion_value double, "
                "assigned_at timestamp)")
    ab, sq = ABTestAnalyzer(con), SequentialTester(con)
    hits = {"fixed": 0, "peek": 0, "old": 0, "paired": 0}
    t0 = pd.Timestamp("2026-01-01")
    for e in range(k):
        eid = f"aa_{e}"
        con.execute("insert into experiments values (?, 'control', 'treatment')", [eid])
        c_all: list[np.ndarray] = []
        t_all: list[np.ndarray] = []
        peek = old_done = paired_done = False
        for look in range(looks):
            n = 2 * per_look
            conv = rng.random(n) < p
            df = pd.DataFrame({
                "experiment_id": eid,
                "variant": np.repeat(["control", "treatment"], per_look),
                "converted": conv, "conversion_value": conv.astype(float),
                "assigned_at": t0 + pd.to_timedelta(np.arange(n) + look * n, unit="s"),
            })
            con.register("batch", df)
            con.execute("insert into experiment_assignments select * from batch")
            con.unregister("batch")
            c_all.append(conv[:per_look].astype(int))
            t_all.append(conv[per_look:].astype(int))

            pv = ab.frequentist_test(eid, metric_type="binary").p_value
            if not peek and pv < ALPHA:
                peek = True
                hits["peek"] += 1
            if look == looks - 1 and pv < ALPHA:
                hits["fixed"] += 1
            if not old_done:
                d = old_sprt_decision(np.concatenate(c_all), np.concatenate(t_all))
                if d != "continue":
                    old_done = True
                    hits["old"] += d == "reject_null"
            if not paired_done:
                d = sq.compute_sprt(eid, alpha=ALPHA, beta=BETA, delta=DELTA).decision
                if d != "continue":
                    paired_done = True
                    hits["paired"] += d == "reject_null"
    print(f"Part 1: {k:,} A/A experiments, {looks * per_look:,} users per arm, {looks} looks")
    for name, label in [("fixed", "z-test, read once at the end"),
                        ("peek", "z-test, stop at first look with p < 0.05"),
                        ("old", "old SPRT (null re-estimated each look)"),
                        ("paired", "paired SPRT (this repo now)")]:
        print(f"  {label:<44} false positives {hits[name] / k:.1%}")


def part_two() -> None:
    rng = np.random.default_rng(11)
    print("Part 2: paired SPRT checked after every pair")
    for p in (0.03, 0.10, 0.30):
        k, n, rej = 4000, 20000, 0
        for _ in range(k):
            c = (rng.random(n) < p).astype(int)
            t = (rng.random(n) < p).astype(int)
            rej += first_crossing(llr_path(c, t, DELTA), UPPER, LOWER)[0] == "reject_null"
        print(f"  A/A at {p:.0%}, {k:,} experiments x {n:,} pairs: false positives {rej / k:.1%}")
    for n in (2000, 5000, 20000):
        k, rej, stops = 1000, 0, []
        for _ in range(k):
            c = (rng.random(n) < 0.10).astype(int)
            t = (rng.random(n) < 0.10 + DELTA).astype(int)
            d, at = first_crossing(llr_path(c, t, DELTA), UPPER, LOWER)
            rej += d == "reject_null"
            if at is not None:
                stops.append(at)
        print(f"  true 10% -> 12% lift, up to {n:,} pairs: caught {rej / k:.1%}, "
              f"median stop at pair {np.median(stops):,.0f}")


if __name__ == "__main__":
    logger.remove()
    part_one()
    part_two()
