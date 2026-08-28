"""Synthetic product event data generator.

Produces realistic user profiles, browsing sessions, and product events
that mirror patterns observed in real SaaS/e-commerce analytics:

- User activity follows a power-law distribution (20% of users generate ~80% of events).
- Sessions cluster on weekdays with bimodal hourly peaks (10 AM, 8 PM).
- In-session event sequences follow a Markov chain transition matrix.
- Funnel progression enforces temporal ordering; drop-off increases at each step.
- Purchase revenue is lognormally distributed (median ~$55, right-skewed tail).
- Three A/B experiments embed known ground-truth effects for validation.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

from src.config.settings import Settings

# ---------------------------------------------------------------------------
# Canonical event names used across all analytics modules.
# ---------------------------------------------------------------------------
EVENT_NAMES: dict[str, list[str]] = {
    "page_view": [
        "view_homepage", "view_product", "view_pricing", "view_blog",
        "view_docs", "view_profile", "view_settings", "view_dashboard",
    ],
    "click": [
        "click_add_to_cart", "click_signup_cta", "click_upgrade",
        "click_learn_more", "click_download", "click_share",
        "begin_checkout",
    ],
    "signup": ["signup_email", "signup_google", "signup_github"],
    "purchase": ["complete_purchase"],
    "feature_use": [
        "use_search", "use_filter", "use_export", "use_import",
        "use_collaboration", "use_api", "use_integration",
    ],
    "form_submit": ["submit_contact_form", "submit_feedback", "submit_survey"],
    "search": ["search_products", "search_docs", "search_help"],
}

# Funnel event names that require ordered progression.
_FUNNEL_EVENTS = [
    "view_homepage", "view_product", "click_add_to_cart",
    "begin_checkout", "complete_purchase",
]

# Markov transition matrix: P(next event_type | current event_type).
_TRANSITION_MATRIX: dict[str, dict[str, float]] = {
    "page_view": {
        "page_view": 0.30, "click": 0.30, "feature_use": 0.15,
        "search": 0.10, "signup": 0.05, "purchase": 0.05, "form_submit": 0.05,
    },
    "click": {
        "page_view": 0.40, "click": 0.20, "feature_use": 0.15,
        "purchase": 0.10, "form_submit": 0.10, "search": 0.05,
    },
    "feature_use": {
        "page_view": 0.30, "click": 0.20, "feature_use": 0.25,
        "search": 0.10, "form_submit": 0.10, "purchase": 0.05,
    },
    "signup": {
        "page_view": 0.50, "feature_use": 0.20, "click": 0.15,
        "search": 0.10, "form_submit": 0.05,
    },
    "purchase": {
        "page_view": 0.60, "click": 0.15, "feature_use": 0.15,
        "search": 0.05, "form_submit": 0.05,
    },
    "form_submit": {
        "page_view": 0.50, "click": 0.20, "feature_use": 0.15,
        "search": 0.10, "form_submit": 0.05,
    },
    "search": {
        "page_view": 0.40, "click": 0.25, "feature_use": 0.15,
        "search": 0.10, "form_submit": 0.05, "purchase": 0.05,
    },
}


def _uid() -> str:
    return uuid.uuid4().hex[:16]


class EventGenerator:
    """Generate synthetic product analytics data with realistic distributions.

    Parameters
    ----------
    settings : Settings
        Application configuration controlling user counts, date range, and
        distribution parameters.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.rng = np.random.default_rng(settings.random_seed)
        self.fake = Faker()
        Faker.seed(settings.random_seed)

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def generate_users(self) -> pd.DataFrame:
        """Generate user profiles with demographic and acquisition attributes.

        Distributions
        -------------
        - signup_source: organic 40%, paid_search 25%, social 15%, referral 10%, email 10%
        - plan_type: free 60%, starter 20%, pro 15%, enterprise 5%
        - industry: 7 verticals weighted toward SaaS and e-commerce
        - company_size: solo/small/medium/large/enterprise
        - country: US-heavy (50%) with long tail
        - Signup dates: uniform across date range with a mild growth trend
          (later months slightly over-represented).
        """
        n = self.settings.num_users
        logger.info("Generating {} users", n)

        signup_sources = self.rng.choice(
            ["organic", "paid_search", "social", "referral", "email"],
            size=n, p=[0.40, 0.25, 0.15, 0.10, 0.10],
        )
        plan_types = self.rng.choice(
            ["free", "starter", "pro", "enterprise"],
            size=n, p=[0.60, 0.20, 0.15, 0.05],
        )
        industries = self.rng.choice(
            ["saas", "ecommerce", "fintech", "healthcare", "education", "media", "other"],
            size=n, p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08],
        )
        company_sizes = self.rng.choice(
            ["solo", "small", "medium", "large", "enterprise"],
            size=n, p=[0.15, 0.30, 0.25, 0.20, 0.10],
        )
        countries = self.rng.choice(
            ["US", "UK", "DE", "CA", "AU", "IN", "BR", "FR", "JP", "Other"],
            size=n, p=[0.50, 0.12, 0.08, 0.07, 0.05, 0.05, 0.04, 0.04, 0.03, 0.02],
        )

        # Signup dates with growth trend: exponential CDF biases toward later dates.
        start = pd.Timestamp(self.settings.date_start)
        end = pd.Timestamp(self.settings.date_end)
        span_days = (end - start).days
        # Beta(2, 1.4) skews mass toward later dates: a mild growth trend,
        # with roughly 60% of signups landing in the second half of the year.
        day_offsets = (self.rng.beta(2, 1.4, size=n) * span_days).astype(int)
        signup_dates = [start + pd.Timedelta(days=int(d)) for d in day_offsets]

        users = pd.DataFrame({
            "user_id": [_uid() for _ in range(n)],
            "created_at": signup_dates,
            "signup_source": signup_sources,
            "plan_type": plan_types,
            "industry": industries,
            "company_size": company_sizes,
            "country": countries,
            "is_active": True,
            "first_purchase_at": pd.NaT,
            "total_revenue": 0.0,
            "lifetime_events": 0,
        })

        logger.info("Generated {} users spanning {} to {}", n, start.date(), end.date())
        return users

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def generate_sessions(self, users: pd.DataFrame) -> pd.DataFrame:
        """Generate browsing sessions for every user.

        Session count per user is drawn from a negative binomial distribution
        (overdispersed Poisson) with mean = avg_sessions_per_user and
        dispersion r = 2.  Power-law adjustment: the top 20% most-active
        users receive 4x the drawn session count.

        Session start times respect weekday/weekend weighting (2:1) and a
        bimodal hourly distribution peaking at 10 AM and 8 PM.
        """
        logger.info("Generating sessions for {} users", len(users))

        mean_s = self.settings.avg_sessions_per_user
        r = 2.0  # dispersion
        p = r / (r + mean_s)  # NB parameterisation: p = r/(r+mu)
        session_counts = self.rng.negative_binomial(r, p, size=len(users))
        session_counts = np.maximum(session_counts, 1)

        # Power-law boost for top 20%
        threshold = np.percentile(session_counts, 80)
        session_counts[session_counts >= threshold] *= 4

        # Honor settings.target_events: the negative binomial plus the
        # power-law boost overshoots (raw parameters produce ~5M events for
        # 50K users, 2.5x the configured target), so scale the whole session
        # distribution down to land just above target. Scaling preserves the
        # power-law shape; only downscale, so small test configs are
        # unaffected.
        expected_events = (
            session_counts.sum() * self.settings.avg_events_per_session
        )
        scale = 1.06 * self.settings.target_events / max(expected_events, 1)
        if scale < 1.0:
            session_counts = np.maximum(
                np.round(session_counts * scale).astype(int), 1
            )
            logger.info(
                "Scaled sessions by {:.2f} to target ~{} events",
                scale, self.settings.target_events,
            )

        entry_pages = ["view_homepage", "view_product", "view_blog", "view_pricing", "view_docs"]
        entry_weights = np.array([0.50, 0.20, 0.15, 0.10, 0.05])

        device_types = ["desktop", "mobile", "tablet"]
        device_weights = [0.55, 0.35, 0.10]
        browsers = ["Chrome", "Safari", "Firefox", "Edge", "Other"]
        browser_weights = [0.60, 0.20, 0.10, 0.08, 0.02]
        os_list = ["Windows", "macOS", "iOS", "Android", "Linux"]
        os_weights = [0.35, 0.25, 0.20, 0.15, 0.05]

        utm_sources = ["google", "facebook", "twitter", "linkedin", "email_campaign"]
        utm_mediums = ["cpc", "social", "email", "referral"]
        utm_campaigns = ["spring_sale", "product_launch", "newsletter", "retarget"]

        # Churn model: each user has an activity lifetime after which they
        # stop generating sessions. 25% of users are long-lived (retained for
        # the whole observation window); the rest churn with exponentially
        # distributed lifetimes (mean 50 days). Cohort retention then decays
        # like 0.25 + 0.75 * exp(-t/50), which matches the shape of real
        # SaaS retention curves (fast early decay onto a loyal-core floor).
        n_users = len(users)
        is_long_lived = self.rng.random(n_users) < 0.25
        lifetimes = np.where(
            is_long_lived,
            np.inf,
            self.rng.exponential(50.0, size=n_users),
        )

        rows: list[dict[str, Any]] = []
        end_dt = pd.Timestamp(self.settings.date_end)

        for idx, (_, user) in enumerate(users.iterrows()):
            n_sessions = int(session_counts[idx])
            user_start = user["created_at"]
            available_days = (end_dt - user_start).days
            if available_days <= 0:
                available_days = 1
            # Sessions only occur inside the user's active lifetime.
            active_days = int(min(available_days, max(lifetimes[idx], 1)))

            for _ in range(n_sessions):
                # Pick a day with weekday bias (Mon-Fri 2x more likely).
                day_offset = self.rng.integers(0, active_days)
                session_day = user_start + pd.Timedelta(days=int(day_offset))
                dow = session_day.dayofweek
                # Reject weekend days with 50% probability to achieve ~2:1 ratio.
                if dow >= 5 and self.rng.random() < 0.50:
                    day_offset = self.rng.integers(0, active_days)
                    session_day = user_start + pd.Timedelta(days=int(day_offset))

                # Bimodal hour: mixture of N(10,2) and N(20,2).
                if self.rng.random() < 0.6:
                    hour = int(np.clip(self.rng.normal(10, 2), 0, 23))
                else:
                    hour = int(np.clip(self.rng.normal(20, 2), 0, 23))
                minute = int(self.rng.integers(0, 60))

                started_at = session_day.replace(hour=hour, minute=minute, second=0)
                if started_at < user_start:
                    started_at = user_start + pd.Timedelta(minutes=int(self.rng.integers(1, 60)))

                device = self.rng.choice(device_types, p=device_weights)
                entry = self.rng.choice(entry_pages, p=entry_weights)

                # UTM tracking: ~40% of sessions have UTM params.
                utm_s: str | None = None
                utm_m: str | None = None
                utm_c: str | None = None
                if self.rng.random() < 0.40:
                    utm_s = str(self.rng.choice(utm_sources))
                    utm_m = str(self.rng.choice(utm_mediums))
                    utm_c = str(self.rng.choice(utm_campaigns))

                rows.append({
                    "session_id": _uid(),
                    "user_id": user["user_id"],
                    "started_at": started_at,
                    "ended_at": None,          # filled after event generation
                    "duration_seconds": None,
                    "event_count": 0,
                    "page_view_count": 0,
                    "has_conversion": False,
                    "entry_page": entry,
                    "exit_page": None,
                    "device_type": device,
                    "browser": self.rng.choice(browsers, p=browser_weights),
                    "os": self.rng.choice(os_list, p=os_weights),
                    "utm_source": utm_s,
                    "utm_medium": utm_m,
                    "utm_campaign": utm_c,
                })

        sessions = pd.DataFrame(rows)
        logger.info("Generated {} sessions", len(sessions))
        return sessions

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def generate_events(
        self,
        sessions: pd.DataFrame,
        users: pd.DataFrame,
        sink: Callable[[pd.DataFrame], object] | None = None,
        chunk_size: int = 250_000,
    ) -> pd.DataFrame:
        """Generate in-session events following a Markov chain.

        Each session starts with a page_view of its entry page. Subsequent
        events are sampled from the transition matrix, with inter-event gaps
        drawn from Exp(mean=45s).

        Funnel behavior is planned, not left to chain luck: every session
        draws its deepest funnel stage from a calibrated distribution and
        those steps (view_homepage ... complete_purchase) are woven into
        the event sequence in order. See the inline calibration note.

        Purchase revenue ~ Lognormal(mu=4.0, sigma=1.0), giving median ~$55.
        Device, browser, and OS are inherited from the parent session.

        Memory model
        ------------
        Holding a full year of events in Python (millions of dicts plus a
        DataFrame copy) needs more RAM than a laptop has; an early full-size
        run died mid-load exactly that way. Pass ``sink`` (a callable that
        receives a DataFrame chunk, e.g. ``ingestion.load_events``) and
        events are flushed every ``chunk_size`` rows, keeping peak memory
        flat regardless of scale. User-level aggregates are accumulated
        incrementally across flushes. Without a sink the full DataFrame is
        returned, which is fine for tests and small runs.
        """
        logger.info("Generating events across {} sessions", len(sessions))

        mean_events = self.settings.avg_events_per_session

        all_events: list[dict[str, Any]] = []
        flushed_frames: list[pd.DataFrame] = []
        rev_acc: dict[str, float] = {}
        evt_acc: dict[str, int] = {}
        first_purchase_acc: dict[str, pd.Timestamp] = {}
        flushed_total = 0

        def _flush() -> None:
            """Convert buffered events to a DataFrame and hand it off."""
            nonlocal flushed_total
            if not all_events:
                return
            df = pd.DataFrame(all_events)
            all_events.clear()
            df["revenue"] = pd.to_numeric(df["revenue"])
            for uid, r in df.groupby("user_id")["revenue"].sum().items():
                rev_acc[uid] = rev_acc.get(uid, 0.0) + float(r or 0.0)
            for uid, n in df.groupby("user_id").size().items():
                evt_acc[uid] = evt_acc.get(uid, 0) + int(n)
            purchases = df[df["event_name"] == "complete_purchase"]
            for uid, ts_min in purchases.groupby("user_id")["timestamp"].min().items():
                prev = first_purchase_acc.get(uid)
                if prev is None or ts_min < prev:
                    first_purchase_acc[uid] = ts_min
            flushed_total += len(df)
            if sink is not None:
                sink(df)
                logger.debug("Flushed {} events (total {})", len(df), flushed_total)
            else:
                flushed_frames.append(df)

        user_country = dict(zip(users["user_id"], users["country"]))

        # Pre-build transition arrays for fast sampling.
        trans_types: dict[str, tuple[list[str], list[float]]] = {}
        for src, dests in _TRANSITION_MATRIX.items():
            types = list(dests.keys())
            prob_arr = np.array(list(dests.values()))
            prob_arr /= prob_arr.sum()
            trans_types[src] = (types, prob_arr.tolist())

        # ------------------------------------------------------------------
        # Funnel plan per session.
        #
        # A Markov chain alone essentially never emits the five funnel events
        # in order inside a ~5-event session (the compound probability is
        # ~1e-5), which would leave the dataset with zero purchases. Instead,
        # each session draws how deep into the funnel it goes, and those
        # events are woven into the sequence in order.
        #
        # _FUNNEL_REACH_P[k-1] = P(session reaches step k), k = 1..4.
        # Calibrated so that user-level, session-scoped funnel conversion
        # over a year (with ~20 sessions per user via 1-(1-p)^n) lands at
        # roughly 100 / 65 / 35 / 18 / 9 percent across the five steps,
        # and ~9% of users ever purchase.
        # ------------------------------------------------------------------
        funnel_reach_p = [0.051, 0.0213, 0.0099, 0.0047]
        funnel_types = {
            "view_homepage": "page_view", "view_product": "page_view",
            "click_add_to_cart": "click", "begin_checkout": "click",
            "complete_purchase": "purchase",
        }
        # Generic page views used to replace organic (Markov-proposed) funnel
        # events, so funnel counts stay exactly as planned.
        nonfunnel_pages = [p for p in EVENT_NAMES["page_view"]
                           if p not in _FUNNEL_EVENTS]

        sessions_list = sessions.to_dict("records")

        for sess in sessions_list:
            n_events = max(1, int(self.rng.poisson(mean_events)))
            ts = pd.Timestamp(sess["started_at"])
            session_id = sess["session_id"]
            user_id = sess["user_id"]
            device = sess["device_type"]
            browser = sess["browser"]
            os_name = sess["os"]
            country = user_country.get(user_id, "US")

            # Draw the session's deepest funnel stage (0 = no funnel intent).
            u = self.rng.random()
            deepest = sum(u < p for p in funnel_reach_p)
            funnel_positions: dict[int, int] = {}
            if deepest >= 1:
                # Funnel sessions start on the homepage and need room for
                # every planned step after it.
                sess["entry_page"] = "view_homepage"
                n_events = max(n_events, deepest + 2)
                slots = sorted(self.rng.choice(
                    np.arange(1, n_events), size=deepest, replace=False))
                funnel_positions = {int(pos): fi + 1
                                    for fi, pos in enumerate(slots)}

            funnel_stage = -1  # index into _FUNNEL_EVENTS reached so far
            page_views = 0
            last_page = sess["entry_page"]
            session_events: list[dict[str, Any]] = []

            current_type = "page_view"

            for i in range(n_events):
                if i == 0:
                    # First event is always a page_view of the entry page.
                    etype = "page_view"
                    ename = sess["entry_page"]
                    if ename == "view_homepage":
                        funnel_stage = 0
                elif i in funnel_positions:
                    # Planned funnel step, emitted in order.
                    fi = funnel_positions[i]
                    ename = _FUNNEL_EVENTS[fi]
                    etype = funnel_types[ename]
                    funnel_stage = fi
                else:
                    # Sample next event type from Markov chain.
                    types, probs = trans_types.get(
                        current_type, trans_types["page_view"]
                    )
                    etype = self.rng.choice(types, p=probs)
                    # Pick a specific event name for this type.
                    candidates = EVENT_NAMES.get(etype)
                    if candidates:
                        ename = self.rng.choice(candidates)
                    else:
                        ename = etype
                    # Organic funnel-event proposals are replaced with
                    # generic page views: funnel progression is governed by
                    # the session plan above, never by chain luck.
                    if ename in _FUNNEL_EVENTS:
                        etype = "page_view"
                        ename = self.rng.choice(nonfunnel_pages)

                # Revenue for purchases.
                revenue = None
                if ename == "complete_purchase":
                    revenue = round(float(self.rng.lognormal(4.0, 1.0)), 2)

                if etype == "page_view":
                    page_views += 1
                    last_page = ename

                # Inter-event gap (exponential, mean 45s).
                if i > 0:
                    gap = max(1, int(self.rng.exponential(45)))
                    ts = ts + pd.Timedelta(seconds=gap)

                properties: dict[str, Any] = {}
                if etype == "page_view":
                    properties["page_title"] = ename.replace("view_", "").replace("_", " ").title()

                session_events.append({
                    "event_id": _uid(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "event_type": etype,
                    "event_name": ename,
                    "timestamp": ts,
                    "properties": json.dumps(properties) if properties else None,
                    "page_url": f"/{ename.replace('_', '/')}",
                    "referrer": None,
                    "utm_source": sess["utm_source"],
                    "utm_medium": sess["utm_medium"],
                    "utm_campaign": sess["utm_campaign"],
                    "device_type": device,
                    "browser": browser,
                    "os": os_name,
                    "country": country,
                    "city": None,
                    "revenue": revenue,
                })

                current_type = etype

            # Back-fill session aggregates.
            sess["ended_at"] = ts
            sess["duration_seconds"] = int(
                (ts - pd.Timestamp(sess["started_at"])).total_seconds()
            )
            sess["event_count"] = len(session_events)
            sess["page_view_count"] = page_views
            sess["has_conversion"] = funnel_stage == len(_FUNNEL_EVENTS) - 1
            sess["exit_page"] = last_page

            all_events.extend(session_events)
            if sink is not None and len(all_events) >= chunk_size:
                _flush()

        _flush()
        logger.info("Generated {} events", flushed_total)

        # Write the back-filled aggregates onto the caller's sessions
        # DataFrame. to_dict("records") copies rows, so mutating the dicts
        # above does not touch the original frame; without this write-back
        # every session would load with no end time, zero events, and
        # has_conversion=false.
        sessions_filled = pd.DataFrame(sessions_list)
        for col in ["ended_at", "duration_seconds", "event_count",
                    "page_view_count", "has_conversion", "exit_page"]:
            sessions[col] = sessions_filled[col].values

        # Apply the incrementally accumulated user-level aggregates.
        users["total_revenue"] = users["user_id"].map(rev_acc).fillna(0)
        users["lifetime_events"] = (
            users["user_id"].map(evt_acc).fillna(0).astype(int)
        )
        users["first_purchase_at"] = users["user_id"].map(first_purchase_acc)

        if sink is not None:
            # Events already delivered chunk by chunk; nothing to return.
            return pd.DataFrame()
        return pd.concat(flushed_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    def generate_experiments(
        self,
        users: pd.DataFrame,
        events: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate three A/B experiments with known ground-truth effects.

        Experiment 1 -- checkout_flow_redesign
            Metric: conversion_rate (binary).
            Control 12%, treatment 14% (true lift ~16.7%).
            5 000 users per variant, start 2025-03-01.

        Experiment 2 -- pricing_page_copy
            Metric: revenue_per_user (continuous).
            Control mean=$45 std=$30, treatment mean=$52 std=$35 (lift ~15.6%).
            3 000 users per variant, start 2025-05-01.

        Experiment 3 -- onboarding_tutorial
            Metric: 7-day retention (binary).
            Control 35%, treatment 33% (no real effect, slight negative).
            4 000 users per variant, start 2025-07-01.
        """
        logger.info("Generating 3 A/B experiments")

        experiments_meta = [
            {
                "experiment_id": "checkout_flow_redesign",
                "experiment_name": "Checkout Flow Redesign",
                "description": "Streamlined 2-step checkout vs original 4-step flow",
                "start_date": "2025-03-01",
                "end_date": "2025-06-01",
                "status": "completed",
                "metric_name": "conversion_rate",
                "control_variant": "control",
                "treatment_variant": "treatment",
                "target_sample_size": 10000,
                "minimum_effect_size": 0.02,
            },
            {
                "experiment_id": "pricing_page_copy",
                "experiment_name": "Pricing Page Copy",
                "description": "Value-focused copy vs feature-list copy on pricing page",
                "start_date": "2025-05-01",
                "end_date": "2025-08-01",
                "status": "completed",
                "metric_name": "revenue_per_user",
                "control_variant": "control",
                "treatment_variant": "treatment",
                "target_sample_size": 6000,
                "minimum_effect_size": 0.10,
            },
            {
                "experiment_id": "onboarding_tutorial",
                "experiment_name": "Onboarding Tutorial",
                "description": "Interactive tutorial vs static documentation for new users",
                "start_date": "2025-07-01",
                "end_date": "2025-10-01",
                "status": "completed",
                "metric_name": "7_day_retention",
                "control_variant": "control",
                "treatment_variant": "treatment",
                "target_sample_size": 8000,
                "minimum_effect_size": 0.03,
            },
        ]
        experiments_df = pd.DataFrame(experiments_meta)

        # ---- Assignments ----
        assignment_rows: list[dict[str, Any]] = []
        eligible_ids = users["user_id"].values

        # Experiment 1: binary conversion
        exp1_users = self.rng.choice(eligible_ids, size=min(10000, len(eligible_ids)), replace=False)
        for i, uid in enumerate(exp1_users):
            variant = "control" if i < len(exp1_users) // 2 else "treatment"
            rate = 0.12 if variant == "control" else 0.14
            converted = bool(self.rng.random() < rate)
            assignment_rows.append({
                "user_id": uid,
                "experiment_id": "checkout_flow_redesign",
                "variant": variant,
                "assigned_at": pd.Timestamp("2025-03-01") + pd.Timedelta(days=int(self.rng.integers(0, 90))),
                "converted": converted,
                "conversion_value": 1.0 if converted else 0.0,
            })

        # Experiment 2: continuous revenue
        exp2_users = self.rng.choice(eligible_ids, size=min(6000, len(eligible_ids)), replace=False)
        for i, uid in enumerate(exp2_users):
            variant = "control" if i < len(exp2_users) // 2 else "treatment"
            if variant == "control":
                value = float(max(0, self.rng.normal(45, 30)))
            else:
                value = float(max(0, self.rng.normal(52, 35)))
            assignment_rows.append({
                "user_id": uid,
                "experiment_id": "pricing_page_copy",
                "variant": variant,
                "assigned_at": pd.Timestamp("2025-05-01") + pd.Timedelta(days=int(self.rng.integers(0, 90))),
                "converted": value > 0,
                "conversion_value": round(value, 2),
            })

        # Experiment 3: binary retention (no real effect)
        exp3_users = self.rng.choice(eligible_ids, size=min(8000, len(eligible_ids)), replace=False)
        for i, uid in enumerate(exp3_users):
            variant = "control" if i < len(exp3_users) // 2 else "treatment"
            rate = 0.35 if variant == "control" else 0.33
            converted = bool(self.rng.random() < rate)
            assignment_rows.append({
                "user_id": uid,
                "experiment_id": "onboarding_tutorial",
                "variant": variant,
                "assigned_at": pd.Timestamp("2025-07-01") + pd.Timedelta(days=int(self.rng.integers(0, 90))),
                "converted": converted,
                "conversion_value": 1.0 if converted else 0.0,
            })

        assignments_df = pd.DataFrame(assignment_rows)
        logger.info(
            "Generated {} experiment assignments across {} experiments",
            len(assignments_df),
            len(experiments_df),
        )
        return experiments_df, assignments_df
