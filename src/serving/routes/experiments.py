"""Experiment endpoints — A/B test results, sequential analysis, and power calculations.

Surfaces both frequentist and Bayesian analysis for each experiment, SPRT-based
sequential testing boundaries, and prospective power/sample-size calculations.
"""

from datetime import datetime

import duckdb
from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


class ExperimentSummary(BaseModel):
    """Headline information for an experiment listing."""

    experiment_id: str
    experiment_name: str
    status: str
    metric_name: str
    start_date: str
    end_date: str | None = None
    observed_lift: float | None = None
    p_value: float | None = None


class FrequentistResponse(BaseModel):
    """Frequentist test output (z-test or Welch's t-test)."""

    test_statistic: float
    p_value: float
    confidence_interval: list[float]
    effect_size: float
    statistical_power: float
    is_significant: bool


class BayesianResponse(BaseModel):
    """Bayesian posterior analysis output."""

    prob_treatment_better: float
    expected_lift: float
    lift_credible_interval: list[float]
    expected_loss_treatment: float
    expected_loss_control: float
    recommendation: str


class ExperimentDetailResponse(BaseModel):
    """Full A/B test analysis combining frequentist and Bayesian results."""

    experiment_id: str
    experiment_name: str
    metric_name: str
    control_metric: float
    treatment_metric: float
    observed_lift: float
    frequentist: FrequentistResponse
    bayesian: BayesianResponse
    sample_sizes: dict[str, int]
    computed_at: datetime


class SPRTResponse(BaseModel):
    """Sequential Probability Ratio Test status."""

    current_llr: float
    upper_boundary: float
    lower_boundary: float
    decision: str
    observations_so_far: int
    estimated_observations_remaining: int | None = None


class PowerRequest(BaseModel):
    """Input for prospective sample-size / power calculations."""

    metric_type: str = Field("binary", description="'binary' or 'continuous'.")
    baseline_rate: float | None = Field(None, description="Baseline conversion rate (binary).")
    baseline_mean: float | None = Field(None, description="Baseline mean (continuous).")
    baseline_std: float | None = Field(None, description="Baseline std dev (continuous).")
    minimum_detectable_effect: float = Field(..., description="Relative lift to detect.")
    alpha: float = 0.05
    power: float = 0.80


class PowerResponse(BaseModel):
    """Sample-size calculation output."""

    sample_size_per_variant: int
    total_sample_size: int
    metric_type: str
    minimum_detectable_effect: float
    alpha: float
    power: float


def _get_conn(request: Request) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(request.app.state.db_path, read_only=True)


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments(request: Request) -> list[ExperimentSummary]:
    """List all experiments with status and headline metrics."""
    conn = _get_conn(request)
    try:
        rows = conn.execute(
            "select experiment_id, experiment_name, status, metric_name, "
            "start_date, end_date from experiments order by start_date desc"
        ).fetchall()
        return [
            ExperimentSummary(
                experiment_id=r[0], experiment_name=r[1], status=r[2],
                metric_name=r[3], start_date=str(r[4]),
                end_date=str(r[5]) if r[5] else None,
            )
            for r in rows
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(
    request: Request,
    experiment_id: str = Path(..., description="Experiment identifier."),
) -> ExperimentDetailResponse:
    """Run full frequentist and Bayesian analysis for a single experiment."""
    conn = _get_conn(request)
    try:
        from src.statistics.ab_testing import ABTestAnalyzer

        analyzer = ABTestAnalyzer(conn)
        result = analyzer.analyze_experiment(experiment_id)
        return ExperimentDetailResponse(
            experiment_id=result.experiment_id,
            experiment_name=result.experiment_name,
            metric_name=result.metric_name,
            control_metric=result.control_metric,
            treatment_metric=result.treatment_metric,
            observed_lift=result.observed_lift,
            frequentist=FrequentistResponse(
                test_statistic=result.frequentist.test_statistic,
                p_value=result.frequentist.p_value,
                confidence_interval=list(result.frequentist.confidence_interval),
                effect_size=result.frequentist.effect_size,
                statistical_power=result.frequentist.statistical_power,
                is_significant=result.frequentist.is_significant,
            ),
            bayesian=BayesianResponse(
                prob_treatment_better=result.bayesian.prob_treatment_better,
                expected_lift=result.bayesian.expected_lift,
                lift_credible_interval=list(result.bayesian.lift_credible_interval),
                expected_loss_treatment=result.bayesian.expected_loss_treatment,
                expected_loss_control=result.bayesian.expected_loss_control,
                recommendation=result.bayesian.recommendation,
            ),
            sample_sizes=result.sample_sizes,
            computed_at=result.computed_at,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.get("/{experiment_id}/sequential", response_model=SPRTResponse)
async def get_sequential_test(
    request: Request,
    experiment_id: str = Path(..., description="Experiment identifier."),
) -> SPRTResponse:
    """Compute SPRT boundaries and current decision status for an experiment."""
    conn = _get_conn(request)
    try:
        from src.statistics.sequential import SequentialTester

        tester = SequentialTester(conn)
        result = tester.compute_sprt(experiment_id)
        return SPRTResponse(
            current_llr=result.current_llr,
            upper_boundary=result.upper_boundary,
            lower_boundary=result.lower_boundary,
            decision=result.decision,
            observations_so_far=result.observations_so_far,
            estimated_observations_remaining=result.estimated_observations_remaining,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@router.post("/power", response_model=PowerResponse)
async def compute_power(request: Request, body: PowerRequest) -> PowerResponse:
    """Compute required sample size per variant for experiment planning."""
    try:
        from src.statistics.power_analysis import PowerAnalyzer

        if body.metric_type == "binary":
            if body.baseline_rate is None:
                raise HTTPException(status_code=400, detail="baseline_rate required for binary metrics.")
            n = PowerAnalyzer.sample_size_binary(
                baseline_rate=body.baseline_rate,
                minimum_detectable_effect=body.minimum_detectable_effect,
                alpha=body.alpha,
                power=body.power,
            )
        else:
            if body.baseline_mean is None or body.baseline_std is None:
                raise HTTPException(
                    status_code=400,
                    detail="baseline_mean and baseline_std required for continuous metrics.",
                )
            n = PowerAnalyzer.sample_size_continuous(
                baseline_mean=body.baseline_mean,
                baseline_std=body.baseline_std,
                minimum_detectable_effect=body.minimum_detectable_effect,
                alpha=body.alpha,
                power=body.power,
            )

        return PowerResponse(
            sample_size_per_variant=n,
            total_sample_size=n * 2,
            metric_type=body.metric_type,
            minimum_detectable_effect=body.minimum_detectable_effect,
            alpha=body.alpha,
            power=body.power,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
