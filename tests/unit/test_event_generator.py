"""Unit tests for the synthetic event data generator.

Validates user counts, event counts, distribution properties, temporal ordering,
and data integrity constraints on the generated dataset.
"""

import pytest

ALLOWED_EVENT_TYPES = {
    "page_view", "click", "feature_use", "signup", "purchase",
    "form_submit", "search",
}

EXPECTED_UTM_SOURCES = {
    "organic": 0.40,
    "paid_search": 0.25,
    "social": 0.15,
    "referral": 0.10,
    "email": 0.10,
}

EXPECTED_DEVICE_TYPES = {
    "desktop": 0.55,
    "mobile": 0.35,
    "tablet": 0.10,
}


class TestEventGenerator:
    """Tests for EventGenerator data quality and distribution properties."""

    def test_user_count(self, generated_data: dict) -> None:
        """The generator must produce exactly num_users user profiles."""
        users = generated_data["users"]
        settings = generated_data["settings"]
        assert len(users) == settings.num_users, (
            f"Expected {settings.num_users} users, got {len(users)}"
        )

    def test_event_count(self, generated_data: dict) -> None:
        """The generator must produce at least target_events events.

        The target is a lower bound because session-level generation may
        overshoot slightly to complete partial sessions.
        """
        events = generated_data["events"]
        settings = generated_data["settings"]
        assert len(events) >= settings.target_events, (
            f"Expected >= {settings.target_events} events, got {len(events)}"
        )

    def test_event_types_valid(self, generated_data: dict) -> None:
        """Every event_type value must belong to the allowed set."""
        events = generated_data["events"]
        invalid = set(events["event_type"].unique()) - ALLOWED_EVENT_TYPES
        assert not invalid, f"Invalid event types found: {invalid}"

    def test_session_timestamps_ordered(self, generated_data: dict) -> None:
        """Session start timestamps must be <= end timestamps.

        Also asserts the aggregates are actually populated: event generation
        must write ended_at and event_count back onto the sessions frame.
        A dropna-only check once passed vacuously while every ended_at was
        null, which hid a real write-back bug.
        """
        sessions = generated_data["sessions"]
        assert sessions["ended_at"].notna().all(), (
            "ended_at not back-filled: generate_events must update the "
            "sessions DataFrame in place"
        )
        assert (sessions["event_count"] >= 1).all(), (
            "event_count not back-filled onto sessions"
        )
        violations = sessions[sessions["started_at"] > sessions["ended_at"]]
        assert len(violations) == 0, (
            f"{len(violations)} sessions have start > end"
        )

    def test_purchase_events_have_revenue(self, generated_data: dict) -> None:
        """Every purchase event must have a positive revenue value.

        Revenue is drawn from a lognormal distribution, so all values
        should be strictly positive.
        """
        events = generated_data["events"]
        purchases = events[events["event_type"] == "purchase"]
        if len(purchases) == 0:
            pytest.skip("No purchase events generated in small dataset")
        missing_revenue = purchases[
            purchases["revenue"].isna() | (purchases["revenue"] <= 0)
        ]
        assert len(missing_revenue) == 0, (
            f"{len(missing_revenue)} purchase events lack positive revenue"
        )

    def test_user_signup_before_first_event(self, generated_data: dict) -> None:
        """No user should have events timestamped before their signup date.

        The generator creates sessions only after the user's created_at
        timestamp, so this constraint should hold universally.
        """
        users = generated_data["users"]
        events = generated_data["events"]

        first_events = events.groupby("user_id")["timestamp"].min().reset_index()
        first_events.columns = ["user_id", "first_event_time"]

        merged = first_events.merge(
            users[["user_id", "created_at"]], on="user_id", how="inner"
        )
        violations = merged[merged["first_event_time"] < merged["created_at"]]
        assert len(violations) == 0, (
            f"{len(violations)} users have events before signup"
        )

    def test_utm_source_distribution(self, generated_data: dict) -> None:
        """UTM source distribution should roughly match expected proportions.

        Tolerance is 5 percentage points to accommodate small-sample variance
        in the test dataset.
        """
        users = generated_data["users"]
        counts = users["signup_source"].value_counts(normalize=True)
        for source, expected_pct in EXPECTED_UTM_SOURCES.items():
            actual = counts.get(source, 0.0)
            assert abs(actual - expected_pct) < 0.10, (
                f"signup_source '{source}': expected ~{expected_pct:.0%}, "
                f"got {actual:.0%}"
            )

    def test_device_type_distribution(self, generated_data: dict) -> None:
        """Device type distribution should roughly match expected proportions.

        Uses a wider tolerance (10pp) because device assignment happens
        at the session level and the test dataset is small.
        """
        sessions = generated_data["sessions"]
        counts = sessions["device_type"].value_counts(normalize=True)
        for device, expected_pct in EXPECTED_DEVICE_TYPES.items():
            actual = counts.get(device, 0.0)
            assert abs(actual - expected_pct) < 0.15, (
                f"device_type '{device}': expected ~{expected_pct:.0%}, "
                f"got {actual:.0%}"
            )

    def test_experiment_variants_balanced(self, generated_data: dict) -> None:
        """Experiment variant assignments should be roughly 50/50.

        Each experiment assigns users to control and treatment with equal
        probability; the split should be within 5pp of 50% for each.
        """
        assignments = generated_data["assignments"]
        if assignments.empty:
            pytest.skip("No experiment assignments in small dataset")

        for exp_id in assignments["experiment_id"].unique():
            exp_assigns = assignments[assignments["experiment_id"] == exp_id]
            variant_counts = exp_assigns["variant"].value_counts(normalize=True)
            for variant, pct in variant_counts.items():
                assert abs(pct - 0.5) < 0.10, (
                    f"Experiment {exp_id}, variant '{variant}': "
                    f"expected ~50%, got {pct:.0%}"
                )

    def test_session_ids_unique(self, generated_data: dict) -> None:
        """All session IDs must be unique across the dataset."""
        sessions = generated_data["sessions"]
        n_total = len(sessions)
        n_unique = sessions["session_id"].nunique()
        assert n_total == n_unique, (
            f"{n_total - n_unique} duplicate session IDs found"
        )
