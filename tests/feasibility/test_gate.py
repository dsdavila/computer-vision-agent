"""Feasibility gate tests — §8 rules, catalog-version discipline, verdict
shape."""

from __future__ import annotations

import pytest

from cv_agent.feasibility import (
    FeasibilityOutcome,
    default_thresholds,
    feasibility,
)
from cv_agent.schemas import (
    AppearanceSeparability,
    Congestion,
    DeploymentHardware,
    HardwareEnvelope,
    Lighting,
    MotionDynamics,
    Ontology,
    OperatingBias,
    OperatingPoint,
    PixelsOnTarget,
    ProbeConfidence,
    ROIPredicate,
    SceneProperties,
    ScoringMetric,
    SuccessCriterion,
    TargetNovelty,
    TaskContract,
)


CATALOG = "v0-2026-09-17"


# --- fixtures ---------------------------------------------------------------


def _scene(
    *,
    p50_px: float = 42.0,
    p50_confidence: ProbeConfidence = ProbeConfidence.HIGH,
    detections_per_frame: float = 6.0,
    occlusion_rate: float = 0.15,
    congestion_confidence: ProbeConfidence = ProbeConfidence.HIGH,
    motion_confidence: ProbeConfidence = ProbeConfidence.HIGH,
    lighting_confidence: ProbeConfidence = ProbeConfidence.HIGH,
    separability_confidence: ProbeConfidence = ProbeConfidence.MEDIUM,
    novelty_confidence: ProbeConfidence = ProbeConfidence.HIGH,
    catalog_version: str = CATALOG,
) -> SceneProperties:
    return SceneProperties(
        pixels_on_target=PixelsOnTarget(
            p10_px=max(0.0, p50_px - 8.0),
            p50_px=p50_px,
            confidence=p50_confidence,
        ),
        congestion=Congestion(
            detections_per_frame_mean=detections_per_frame,
            mean_pairwise_iou=0.08,
            occlusion_rate=occlusion_rate,
            confidence=congestion_confidence,
        ),
        motion_dynamics=MotionDynamics(
            optical_flow_mag_p50=1.4,
            per_track_displacement_variance=0.9,
            camera_motion_estimate=0.02,
            confidence=motion_confidence,
        ),
        lighting=Lighting(
            intensity_variance_over_time=310.5,
            saturation_clipping_fraction=0.01,
            confidence=lighting_confidence,
        ),
        appearance_separability=AppearanceSeparability(
            inter_instance_distance_p50=0.62,
            confidence=separability_confidence,
        ),
        target_novelty=TargetNovelty(
            open_vocab_score_p10=0.4,
            open_vocab_score_p50=0.7,
            confidence=novelty_confidence,
        ),
        catalog_version=catalog_version,
    )


def _contract(catalog_version: str = CATALOG) -> TaskContract:
    return TaskContract(
        name="test",
        ontology=Ontology(closed_set_classes=("person", "vehicle")),
        predicates=(
            ROIPredicate(
                name="area",
                polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                target_classes=("person",),
            ),
        ),
        operating_point=OperatingPoint(bias=OperatingBias.BALANCED),
        deployment_hardware=DeploymentHardware(
            hardware=HardwareEnvelope(gpu_model="x", vram_gb=8, cpu_cores=4),
            latency_budget_ms=100.0,
        ),
        success_criteria=(
            SuccessCriterion(
                metric=ScoringMetric.RECALL,
                per_class="person",
                min_value=0.8,
            ),
        ),
        catalog_version=catalog_version,
    )


# --- happy path -------------------------------------------------------------


def test_healthy_scene_proceeds():
    verdict = feasibility(_scene(), _contract(), default_thresholds(CATALOG))
    assert verdict.outcome is FeasibilityOutcome.PROCEED
    assert verdict.reasons == ()
    assert verdict.passed and not verdict.refused
    assert verdict.catalog_version == CATALOG


# --- pixels-on-target floor -------------------------------------------------


def test_tiny_targets_refuse_with_physical_reason():
    scene = _scene(p50_px=11.0)  # matches §8's canonical example
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    assert verdict.refused
    rule_ids = {r.rule for r in verdict.reasons}
    assert "pixels_on_target_floor" in rule_ids
    reason = next(r for r in verdict.reasons if r.rule == "pixels_on_target_floor")
    # The message quotes the actual measurement and the threshold, so a human
    # can act on it.
    assert "11.0" in reason.message
    assert "20" in reason.message


def test_targets_exactly_at_floor_pass():
    # 20.0 == floor; not below → not refused by this rule.
    scene = _scene(p50_px=20.0)
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    assert "pixels_on_target_floor" not in {r.rule for r in verdict.reasons}


# --- congestion ceiling ----------------------------------------------------


def test_high_density_refuses():
    scene = _scene(detections_per_frame=80.0)
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    assert verdict.refused
    reason = next(r for r in verdict.reasons if r.rule == "congestion_ceiling")
    assert "detection density" in reason.message
    assert "80" in reason.message


def test_high_occlusion_refuses():
    scene = _scene(occlusion_rate=0.85)
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    reason = next(r for r in verdict.reasons if r.rule == "congestion_ceiling")
    assert "occlusion rate" in reason.message
    assert "0.85" in reason.message


def test_both_congestion_bounds_named_in_one_reason():
    scene = _scene(detections_per_frame=80.0, occlusion_rate=0.85)
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    reasons = [r for r in verdict.reasons if r.rule == "congestion_ceiling"]
    assert len(reasons) == 1
    assert "detection density" in reasons[0].message
    assert "occlusion rate" in reasons[0].message


# --- low-profiling-confidence refusal ---------------------------------------


def test_low_confidence_anywhere_refuses():
    scene = _scene(lighting_confidence=ProbeConfidence.LOW)
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    assert verdict.refused
    rule_ids = {r.rule for r in verdict.reasons}
    assert "low_profiling_confidence" in rule_ids
    reason = next(r for r in verdict.reasons if r.rule == "low_profiling_confidence")
    assert "lighting" in reason.message


def test_low_confidence_hides_other_rules_for_the_low_axis():
    # A LOW pixels_on_target should NOT double-refuse under both
    # pixels_on_target_floor and low_profiling_confidence — the floor rule
    # skips low-confidence readings, only the confidence rule fires.
    scene = _scene(p50_px=5.0, p50_confidence=ProbeConfidence.LOW)
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    rules = [r.rule for r in verdict.reasons]
    assert "low_profiling_confidence" in rules
    assert "pixels_on_target_floor" not in rules


def test_multiple_low_axes_named_in_one_reason():
    scene = _scene(
        lighting_confidence=ProbeConfidence.LOW,
        motion_confidence=ProbeConfidence.LOW,
    )
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    reason = next(r for r in verdict.reasons if r.rule == "low_profiling_confidence")
    assert "lighting" in reason.message
    assert "motion_dynamics" in reason.message


# --- catalog-version discipline --------------------------------------------


def test_catalog_mismatch_between_scene_and_contract_raises():
    scene = _scene(catalog_version="v0-2026-11-01")
    contract = _contract(catalog_version=CATALOG)
    with pytest.raises(ValueError, match="catalog_version mismatch"):
        feasibility(scene, contract, default_thresholds(CATALOG))


def test_catalog_mismatch_between_scene_and_thresholds_raises():
    scene = _scene(catalog_version="v0-2026-11-01")
    with pytest.raises(ValueError, match="catalog_version mismatch"):
        feasibility(
            scene,
            _contract(catalog_version="v0-2026-11-01"),
            default_thresholds(CATALOG),
        )


def test_default_thresholds_unknown_catalog_raises():
    with pytest.raises(ValueError, match="No feasibility thresholds pinned"):
        default_thresholds("v99-invented")


def test_verdict_carries_thresholds_catalog_version():
    verdict = feasibility(_scene(), _contract(), default_thresholds(CATALOG))
    assert verdict.catalog_version == CATALOG


# --- multi-rule composition -------------------------------------------------


def test_multiple_rules_fire_together():
    scene = _scene(p50_px=8.0, detections_per_frame=100.0)
    verdict = feasibility(scene, _contract(), default_thresholds(CATALOG))
    rule_ids = {r.rule for r in verdict.reasons}
    assert "pixels_on_target_floor" in rule_ids
    assert "congestion_ceiling" in rule_ids
    # Both are named so the human sees everything wrong at once.
