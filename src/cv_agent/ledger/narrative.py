"""Narrative rendering at decision time — §9 "two renderings, one store".

Deterministic template-based renderer that composes LedgerEntry fields into
a short causal narrative. Written at decision time (before the entry is
finalized) so the rendered string is durable — a human returning in month
seven does not need whatever model is available later to read the record.

Between two interpretations of §9 ("freeze an LLM's output" vs. "generate
deterministically"), templates are the more honest choice:
- Narrative quality does not depend on model quality (§14 on-prem model
  degradation risk row).
- The renderer runs in the deterministic layer, matching §2's commitment
  that the LLM is not on the inner scoring loop.

Signature takes explicit keyword args mirroring LedgerEntry fields rather
than a LedgerEntry directly — narrative is a required field on the entry,
so the entry cannot exist before the narrative is rendered.
"""

from __future__ import annotations

from cv_agent.schemas import (
    ConfigValue,
    MetricReading,
    Prediction,
    ProbeConfidence,
    SceneProperties,
    ScorerTier,
    Verdict,
)


def render_narrative(
    *,
    changed_parameters: tuple[str, ...],
    full_config: dict[str, ConfigValue],
    hypothesis: tuple[Prediction, ...],
    outcome: tuple[MetricReading, ...],
    verdict: Verdict,
    scene_properties: SceneProperties,
    scorer_tier: ScorerTier,
    confounded: bool,
) -> str:
    """Produce the narrative string for a LedgerEntry.

    Kwargs mirror the LedgerEntry fields the narrative reads. The caller
    passes them explicitly, uses the returned string as the entry's
    `narrative` field, then constructs the entry."""

    parts = [
        _render_change(changed_parameters, full_config),
        _render_hypothesis_outcome(hypothesis, outcome, verdict),
        _render_scene(scene_properties),
        _render_caveats(confounded, scene_properties),
        _render_tier(scorer_tier),
    ]
    return " ".join(part for part in parts if part)


# --- section renderers ------------------------------------------------------


def _render_change(
    changed: tuple[str, ...], config: dict[str, ConfigValue]
) -> str:
    if not changed:
        return "New root config."
    pairs = ", ".join(f"{p}={config[p]!r}" for p in changed)
    if len(changed) == 1:
        return f"Changed {pairs}."
    return f"Changed {len(changed)} parameters: {pairs}."


def _render_hypothesis_outcome(
    hypothesis: tuple[Prediction, ...],
    outcome: tuple[MetricReading, ...],
    verdict: Verdict,
) -> str:
    """Match predictions to outcomes by (metric_name, stratum) so the
    narrative names the measured Δ where available."""
    outcomes_by_key = {(r.name, r.stratum): r for r in outcome}
    fragments: list[str] = []
    for pred in hypothesis:
        actual = outcomes_by_key.get((pred.metric_name, pred.stratum))
        scope = f"[{pred.stratum}]" if pred.stratum else ""
        pred_str = (
            f"predicted {pred.metric_name}{scope} "
            f"Δ={pred.predicted_delta:+.3f} (±{pred.tolerance:.3f})"
        )
        if actual is not None:
            measured_delta = actual.value - pred.baseline_value
            pred_str += f"; measured Δ={measured_delta:+.3f}"
        fragments.append(pred_str)
    if not fragments:
        return f"Verdict: {verdict.value}."
    return f"Hypothesis: {'; '.join(fragments)}. Verdict: {verdict.value}."


def _render_scene(scene: SceneProperties) -> str:
    p = scene.pixels_on_target
    c = scene.congestion
    m = scene.motion_dynamics
    l = scene.lighting  # noqa: E741 — short, clear in context
    a = scene.appearance_separability
    n = scene.target_novelty
    return (
        "Scene: "
        f"target p50 {p.p50_px:.1f}px ({p.confidence.value}); "
        f"{c.detections_per_frame_mean:.1f}/frame, "
        f"{c.occlusion_rate:.2f} occlusion ({c.confidence.value}); "
        f"motion flow p50 {m.optical_flow_mag_p50:.2f} ({m.confidence.value}); "
        f"lighting variance {l.intensity_variance_over_time:.1f} ({l.confidence.value}); "
        f"separability {a.inter_instance_distance_p50:.2f} ({a.confidence.value}); "
        f"target-novelty p50 {n.open_vocab_score_p50:.2f} ({n.confidence.value})."
    )


def _render_caveats(confounded: bool, scene: SceneProperties) -> str:
    caveats: list[str] = []
    if confounded:
        caveats.append(
            "Confounded: multi-variable or joint-refinement delta; causal attribution not established (§9)."
        )
    low_axes = [
        name
        for name, reading in (
            ("pixels_on_target", scene.pixels_on_target),
            ("congestion", scene.congestion),
            ("motion_dynamics", scene.motion_dynamics),
            ("lighting", scene.lighting),
            ("appearance_separability", scene.appearance_separability),
            ("target_novelty", scene.target_novelty),
        )
        if reading.confidence is ProbeConfidence.LOW
    ]
    if low_axes:
        caveats.append(
            f"LOW confidence on: {', '.join(low_axes)} (§4.1 — regime classification may be unreliable)."
        )
    return " ".join(caveats)


def _render_tier(tier: ScorerTier) -> str:
    caveats = {
        ScorerTier.HARD_CONSTRAINT: "hard-constraint gate; filtering only, no score.",
        ScorerTier.LABEL_FREE_PROXY: "label-free proxy; not orderable across topologies (§6).",
        ScorerTier.VLM_PAIRWISE: "VLM pairwise preference; use to prune, not decide (§6).",
        ScorerTier.GROUND_TRUTH: "ground-truth eval set; per-stratum CIs (§7.2).",
    }
    return f"Tier {tier.value}: {caveats[tier]}"
