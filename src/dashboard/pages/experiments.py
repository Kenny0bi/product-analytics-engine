"""Experiments page: the anatomy of a decision.

Three linked views per experiment, each answering a different question:

1. Posterior landscape: what do we now believe each variant's true rate is?
   Two Beta (or Normal) posteriors drawn as overlapping hills; the overlap
   is the remaining uncertainty about which is better.
2. The decision corridor: the SPRT log-likelihood ratio walking between
   Wald's boundaries as observations accumulate. Crossing the top rail is
   evidence for the treatment; the bottom rail, evidence of no effect.
3. The power curve: what this experiment's sample size could and could not
   have detected in the first place.

The verdict card on top translates all of it into one plain sentence.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src.dashboard import theme as T
from src.statistics.ab_testing import ABTestAnalyzer
from src.statistics.power_analysis import PowerAnalyzer
from src.statistics.sequential import SequentialTester


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _posterior_chart(stats_by_variant: dict, metric_type: str) -> go.Figure:
    """Overlapping posterior densities for control and treatment."""
    fig = T.base_figure(height=340)
    colors = {"control": T.INK_MUTED, "treatment": T.VERMILION}

    # Common x-range across both posteriors.
    params: dict[str, tuple[str, float, float]] = {}
    for variant, s in stats_by_variant.items():
        n = int(s["n"])
        if metric_type == "binary":
            x = int(s["conversions"])
            params[variant] = ("beta", 1 + x, 1 + n - x)
        else:
            mean = float(s["mean_value"] or 0)
            sem = float(s["std_value"] or 1) / math.sqrt(max(n, 1))
            params[variant] = ("norm", mean, sem)

    lo = min(stats.beta.ppf(0.0005, p[1], p[2]) if p[0] == "beta"
             else stats.norm.ppf(0.0005, p[1], p[2]) for p in params.values())
    hi = max(stats.beta.ppf(0.9995, p[1], p[2]) if p[0] == "beta"
             else stats.norm.ppf(0.9995, p[1], p[2]) for p in params.values())
    xs = np.linspace(lo, hi, 400)

    for variant, p in params.items():
        pdf = (stats.beta.pdf(xs, p[1], p[2]) if p[0] == "beta"
               else stats.norm.pdf(xs, p[1], p[2]))
        mean = (p[1] / (p[1] + p[2])) if p[0] == "beta" else p[1]
        color = colors.get(variant, T.BLUE)
        fmt = f"{mean:.2%}" if metric_type == "binary" else f"${mean:,.2f}"
        fig.add_trace(go.Scatter(
            x=xs, y=pdf, mode="lines", name=variant,
            line=dict(color=color, width=2.4),
            fill="tozeroy", fillcolor=_rgba(color, 0.18),
            hovertemplate=(variant + " rate %{x:.3%}<extra></extra>"
                           if metric_type == "binary"
                           else variant + " mean $%{x:,.2f}<extra></extra>"),
        ))
        fig.add_annotation(
            x=mean, y=float(np.max(pdf)) * 1.04, text=f"{variant} {fmt}",
            showarrow=False,
            font=dict(family=T.FONT_MONO, size=11, color=color),
        )

    fig.update_xaxes(
        title="true conversion rate" if metric_type == "binary"
        else "true mean value",
        tickformat=".1%" if metric_type == "binary" else "$,.0f",
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False)
    return fig


def _sprt_chart(conn, experiment_id: str, sprt) -> go.Figure:
    """The LLR trajectory between Wald's decision boundaries.

    Recomputes the cumulative log-likelihood ratio per treatment
    observation with the same likelihood model as SequentialTester, so
    the path shown ends exactly at the tester's reported LLR.
    """
    rows = conn.execute("""
        select ea.variant, case when ea.converted then 1 else 0 end
        from experiment_assignments ea
        where ea.experiment_id = ?
        order by ea.assigned_at
    """, [experiment_id]).fetchall()

    control = [r[1] for r in rows if r[0] == "control"]
    treatment = [r[1] for r in rows if r[0] == "treatment"]
    theta_0 = min(max(float(np.mean(control)), 1e-10), 1 - 1e-10)
    theta_1 = min(max(theta_0 + sprt.delta, 1e-10), 1 - 1e-10)
    lr1 = math.log(theta_1 / theta_0)
    lr0 = math.log((1 - theta_1) / (1 - theta_0))
    llr_path = np.cumsum([x * lr1 + (1 - x) * lr0 for x in treatment])
    ns = np.arange(1, len(llr_path) + 1)

    fig = T.base_figure(height=340)
    upper, lower = sprt.upper_boundary, sprt.lower_boundary
    pad = (upper - lower) * 0.45
    x_max = len(llr_path)

    # Decision regions.
    fig.add_hrect(y0=upper, y1=upper + pad, fillcolor=_rgba(T.TEAL, 0.12),
                  line_width=0)
    fig.add_hrect(y0=lower - pad, y1=lower, fillcolor=_rgba(T.CRIMSON, 0.10),
                  line_width=0)
    fig.add_hline(y=upper, line=dict(color=T.TEAL, width=1.6, dash="dash"))
    fig.add_hline(y=lower, line=dict(color=T.CRIMSON, width=1.6, dash="dash"))
    fig.add_annotation(x=x_max * 0.01, y=upper + pad * 0.5, xanchor="left",
                       text="evidence for the treatment (reject H0)",
                       showarrow=False,
                       font=dict(family=T.FONT_UI, size=11, color=T.GOOD))
    fig.add_annotation(x=x_max * 0.01, y=lower - pad * 0.5, xanchor="left",
                       text="evidence of no effect (accept H0)",
                       showarrow=False,
                       font=dict(family=T.FONT_UI, size=11, color=T.SERIOUS))

    fig.add_trace(go.Scatter(
        x=ns, y=llr_path, mode="lines", name="cumulative LLR",
        line=dict(color=T.INK, width=2),
        hovertemplate="after %{x:,} users: LLR %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[ns[-1]], y=[llr_path[-1]], mode="markers",
        marker=dict(size=10, color=T.INK, line=dict(color=T.PAPER, width=2)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_annotation(
        x=ns[-1], y=llr_path[-1],
        text=f"now: {llr_path[-1]:.2f}", showarrow=True, arrowhead=0,
        ax=-10, ay=-22, arrowcolor=T.INK_MUTED,
        font=dict(family=T.FONT_MONO, size=11, color=T.INK),
    )

    fig.update_xaxes(title="treatment users observed")
    fig.update_yaxes(title="log-likelihood ratio")
    fig.update_layout(showlegend=False)
    return fig


def _power_chart(baseline: float, observed_lift: float,
                 current_n: int) -> go.Figure:
    """Power as a function of per-variant sample size for the observed lift."""
    lift = max(abs(observed_lift), 0.005)
    ns = np.unique(np.geomspace(100, max(current_n * 4, 2000), 60).astype(int))
    powers = [PowerAnalyzer.compute_power(int(n), baseline, lift) for n in ns]

    fig = T.base_figure(height=320)
    fig.add_hline(y=0.80, line=dict(color=T.LINE_SOFT, width=1.4, dash="dot"))
    fig.add_annotation(x=float(ns[2]), y=0.815, text="80% convention",
                       showarrow=False, xanchor="left",
                       font=dict(family=T.FONT_MONO, size=10,
                                 color=T.INK_FAINT))
    fig.add_trace(go.Scatter(
        x=ns, y=powers, mode="lines", name="power",
        line=dict(color=T.PLUM, width=2.4),
        hovertemplate="n=%{x:,}/variant: power %{y:.0%}<extra></extra>",
    ))
    cur_power = PowerAnalyzer.compute_power(current_n, baseline, lift)
    fig.add_trace(go.Scatter(
        x=[current_n], y=[cur_power], mode="markers",
        marker=dict(size=11, color=T.PLUM, line=dict(color=T.PAPER, width=2)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_annotation(
        x=current_n, y=cur_power,
        text=f"this experiment: n={T.fmt_compact(current_n)}, "
             f"power {cur_power:.0%}",
        showarrow=True, arrowhead=0, ax=-8, ay=-26, arrowcolor=T.INK_MUTED,
        font=dict(family=T.FONT_MONO, size=11, color=T.INK),
    )
    fig.update_xaxes(title="users per variant", type="log")
    fig.update_yaxes(title="power to detect the observed lift",
                     tickformat=".0%", range=[0, 1.05])
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render(conn) -> None:
    """Render the experiments page."""
    T.inject_css(st)
    if conn is None:
        st.error("No database found. Run `python -m src.main generate` first.")
        return

    T.headline(
        st, "Experimentation", "The anatomy of a decision",
        "Each experiment gets read three ways: what we now believe "
        "(the posteriors), when we could have stopped (the sequential "
        "corridor), and what we were ever able to detect (the power curve).",
    )

    experiments = conn.execute(
        "select experiment_id, experiment_name, metric_name, start_date "
        "from experiments order by start_date").fetchall()
    if not experiments:
        st.warning("No experiments found in the database.")
        return

    exp_display = {f"{name} ({metric})": eid
                   for eid, name, metric, _ in experiments}
    choice = st.selectbox("Experiment", list(exp_display.keys()))
    exp_id = exp_display[choice]

    analyzer = ABTestAnalyzer(conn)
    result = analyzer.analyze_experiment(exp_id)
    metric_type = ("binary" if result.metric_name
                   in {"conversion_rate", "7_day_retention", "retention"}
                   else "continuous")
    is_binary = metric_type == "binary"

    def fmt_metric(v: float) -> str:
        return f"{v:.2%}" if is_binary else f"${v:,.2f}"

    # --- Verdict card ---
    b = result.bayesian
    f = result.frequentist
    if b.prob_treatment_better > 0.95 and f.is_significant:
        verdict_color, verdict = T.GOOD, "Ship the treatment."
    elif b.prob_treatment_better < 0.05 and f.is_significant:
        verdict_color, verdict = T.SERIOUS, "Keep the control."
    elif not f.is_significant and abs(result.observed_lift) < 0.05:
        verdict_color, verdict = T.INK_MUTED, "No detectable effect."
    else:
        verdict_color, verdict = T.WARNING, "Evidence is not conclusive."
    T.verdict_card(
        st, verdict,
        f"Treatment moved <span class='mono'>{result.metric_name}</span> from "
        f"<span class='mono'>{fmt_metric(result.control_metric)}</span> to "
        f"<span class='mono'>{fmt_metric(result.treatment_metric)}</span> "
        f"(<span class='mono'>{result.observed_lift:+.1%}</span> relative). "
        f"p = <span class='mono'>{f.p_value:.4f}</span>, "
        f"P(treatment &gt; control) = "
        f"<span class='mono'>{b.prob_treatment_better:.1%}</span>. "
        f"Bayesian read: {b.recommendation.lower()}.",
        color=verdict_color,
    )

    T.kpi_tiles(st, [
        {"label": "Control", "value": fmt_metric(result.control_metric)},
        {"label": "Treatment", "value": fmt_metric(result.treatment_metric)},
        {"label": "Observed lift", "value": f"{result.observed_lift:+.1%}",
         "direction": "up" if result.observed_lift > 0 else "down",
         "delta": f"95% CrI [{b.lift_credible_interval[0]:+.1%}, "
                  f"{b.lift_credible_interval[1]:+.1%}]"},
        {"label": "p-value", "value": f"{f.p_value:.4f}"},
        {"label": "Power", "value": f"{f.statistical_power:.0%}"},
        {"label": "Sample",
         "value": " / ".join(T.fmt_compact(n)
                             for n in result.sample_sizes.values())},
    ])

    col1, col2 = st.columns(2)
    vs = analyzer._fetch_variant_stats(exp_id)
    with col1:
        st.markdown("**What we now believe** &middot; posterior densities; "
                    "the overlap is the remaining doubt")
        st.plotly_chart(_posterior_chart(vs, metric_type),
                        width='stretch')
    with col2:
        st.markdown("**The power curve** &middot; could this design ever "
                    "have seen the effect?")
        baseline = (result.control_metric if is_binary else 0.5)
        n_treat = list(result.sample_sizes.values())[-1]
        if is_binary:
            st.plotly_chart(
                _power_chart(baseline, result.observed_lift, n_treat),
                width='stretch')
        else:
            st.caption(
                "Power curves here are drawn for binary metrics; this "
                "experiment's continuous metric is covered by the reported "
                f"power of {f.statistical_power:.0%} from Welch's t-test."
            )

    if is_binary:
        T.rule(st)
        st.markdown("**The decision corridor** &middot; sequential probability "
                    "ratio test: the evidence walking between Wald's rails")
        tester = SequentialTester(conn)
        sprt = tester.compute_sprt(exp_id)
        st.plotly_chart(_sprt_chart(conn, exp_id, sprt),
                        width='stretch')
        decision_txt = {
            "reject_null": "crossed the upper rail: the effect is real at "
                           "these error rates, and the test could have "
                           "stopped there",
            "accept_null": "crossed the lower rail: no meaningful effect at "
                           "these error rates",
            "continue": "still inside the corridor: keep collecting data",
        }[sprt.decision]
        st.caption(
            f"alpha = {sprt.alpha}, beta = {sprt.beta}, minimum detectable "
            f"effect = {sprt.delta:.0%} absolute. After "
            f"{T.fmt_int(sprt.observations_so_far)} treatment observations "
            f"the path has {decision_txt}."
        )
