"""Feasibility gate — §8 "say no early with a physical reason".

`feasibility(scene, contract, thresholds)` runs each rule in the starting
set from §8 and assembles a FeasibilityVerdict. A refusal names a specific
rule and delivers the physical narrative in the shape of §8's canonical
example: "target averages 11 px tall; no catalog method achieves useful
recall below ~20 px; change the lens or accept the number."

Catalog-version discipline: the profiler pins a catalog version on
SceneProperties, the TaskContract pins one on itself, and thresholds pin
one on themselves. All three must match; a mismatch is a hard error rather
than a silent proceed.

Starting rules (§8):
- pixels-on-target floor
- congestion ceiling (two independent bounds)
- low-profiling-confidence refusal (§4.1)

Deferred: appearance-separability-for-ReID. Whether ReID is required is a
planner-stage decision (§3); the profiler cannot decide it at this point.
The rule moves to the planner or the ReID stage specialist later.
"""

from __future__ import annotations

from cv_agent.feasibility.thresholds import FeasibilityThresholds
from cv_agent.feasibility.verdict import (
    FeasibilityOutcome,
    FeasibilityVerdict,
    RefusalReason,
)
from cv_agent.schemas import ProbeConfidence, SceneProperties, TaskContract


def feasibility(
    scene_properties: SceneProperties,
    task_contract: TaskContract,
    thresholds: FeasibilityThresholds,
) -> FeasibilityVerdict:
    """Run the §8 feasibility rules and return a verdict.

    Raises ValueError if the three catalog_version pins do not agree — a
    version mismatch means we would be evaluating a scene against thresholds
    derived for a different catalog, which produces an authoritative-looking
    but silently wrong verdict."""

    _check_catalog_versions(scene_properties, task_contract, thresholds)

    reasons: list[RefusalReason] = []
    for rule in _RULES:
        reason = rule(scene_properties, task_contract, thresholds)
        if reason is not None:
            reasons.append(reason)

    outcome = FeasibilityOutcome.PROCEED if not reasons else FeasibilityOutcome.REFUSE
    return FeasibilityVerdict(
        outcome=outcome,
        reasons=tuple(reasons),
        catalog_version=thresholds.catalog_version,
    )


# --- catalog-version discipline --------------------------------------------


def _check_catalog_versions(
    scene: SceneProperties,
    contract: TaskContract,
    thresholds: FeasibilityThresholds,
) -> None:
    versions = {
        "scene_properties.catalog_version": scene.catalog_version,
        "task_contract.catalog_version": contract.catalog_version,
        "thresholds.catalog_version": thresholds.catalog_version,
    }
    unique = set(versions.values())
    if len(unique) > 1:
        pairs = ", ".join(f"{k}={v!r}" for k, v in versions.items())
        raise ValueError(
            f"feasibility: catalog_version mismatch across inputs ({pairs}). "
            "All three must agree or the verdict is not interpretable."
        )


# --- rules ------------------------------------------------------------------
#
# Each rule has signature (scene, contract, thresholds) -> RefusalReason | None.
# The gate collects every reason that fires — we don't short-circuit, because
# the human seeing the refusal wants to know everything wrong with the setup,
# not just the first thing.


def _rule_low_profiling_confidence(
    scene: SceneProperties,
    contract: TaskContract,
    thresholds: FeasibilityThresholds,
) -> RefusalReason | None:
    """§4.1 — no verdict beats a confident-wrong verdict. Any probe at LOW
    triggers refusal so downstream never operates on garbage input."""
    del contract, thresholds
    low_axes: list[str] = []
    for name, reading in [
        ("pixels_on_target", scene.pixels_on_target),
        ("congestion", scene.congestion),
        ("motion_dynamics", scene.motion_dynamics),
        ("lighting", scene.lighting),
        ("appearance_separability", scene.appearance_separability),
        ("target_novelty", scene.target_novelty),
    ]:
        if reading.confidence is ProbeConfidence.LOW:
            low_axes.append(name)
    if not low_axes:
        return None
    joined = ", ".join(low_axes)
    return RefusalReason(
        rule="low_profiling_confidence",
        message=(
            f"Profiling confidence is LOW on: {joined}. §4.1: no verdict "
            "beats a confident-wrong verdict. Rerun with more/better data, "
            "or plug in a working detector for proposal-dependent probes."
        ),
    )


def _rule_pixels_on_target_floor(
    scene: SceneProperties,
    contract: TaskContract,
    thresholds: FeasibilityThresholds,
) -> RefusalReason | None:
    """§8 canonical rule. Refuses when p50 box height is below the catalog-
    derived floor."""
    del contract
    p = scene.pixels_on_target
    # Low-confidence probes are handled by _rule_low_profiling_confidence.
    # Don't double-refuse; and reading a low-confidence value is not honest
    # signal for this rule.
    if p.confidence is ProbeConfidence.LOW:
        return None
    if p.p50_px < thresholds.min_pixels_on_target_p50_px:
        return RefusalReason(
            rule="pixels_on_target_floor",
            message=(
                f"Target averages {p.p50_px:.1f} px tall (p50). "
                f"No catalog method in {thresholds.catalog_version} "
                f"achieves useful recall below ~{thresholds.min_pixels_on_target_p50_px:.0f} px. "
                "Change the lens, get closer, or accept a reduced pipeline."
            ),
        )
    return None


def _rule_congestion_ceiling(
    scene: SceneProperties,
    contract: TaskContract,
    thresholds: FeasibilityThresholds,
) -> RefusalReason | None:
    """§8 — refuses when the scene is too crowded for catalog trackers.
    Two independent bounds; hitting either fires the refusal (the message
    names whichever is worse)."""
    del contract
    c = scene.congestion
    if c.confidence is ProbeConfidence.LOW:
        return None

    density_over = c.detections_per_frame_mean > thresholds.max_congestion_detections_per_frame
    occlusion_over = c.occlusion_rate > thresholds.max_congestion_occlusion_rate
    if not (density_over or occlusion_over):
        return None

    parts: list[str] = []
    if density_over:
        parts.append(
            f"detection density is {c.detections_per_frame_mean:.1f}/frame "
            f"(catalog trackers in {thresholds.catalog_version} handle up to "
            f"~{thresholds.max_congestion_detections_per_frame:.0f})"
        )
    if occlusion_over:
        parts.append(
            f"occlusion rate is {c.occlusion_rate:.2f} "
            f"(catalog trackers degrade above {thresholds.max_congestion_occlusion_rate:.2f})"
        )
    joined = "; ".join(parts)
    return RefusalReason(
        rule="congestion_ceiling",
        message=(
            f"Scene is too congested: {joined}. Reduce the ROI, raise the "
            "camera, or accept identity switches around occlusions."
        ),
    )


_RULES = (
    _rule_low_profiling_confidence,
    _rule_pixels_on_target_floor,
    _rule_congestion_ceiling,
)
