"""LedgerEntry schema tests — §9 fields, §9 invariants."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cv_agent.schemas import (
    AppearanceSeparability,
    Congestion,
    CostMeasurement,
    HardwareEnvelope,
    LedgerEntry,
    Lighting,
    MetricReading,
    MotionDynamics,
    PixelsOnTarget,
    Prediction,
    ProbeConfidence,
    Provenance,
    SceneProperties,
    ScorerTier,
    TargetCostMeasurement,
    TargetNovelty,
    Verdict,
)


# --- fixtures ---------------------------------------------------------------


def _scene() -> SceneProperties:
    return SceneProperties(
        pixels_on_target=PixelsOnTarget(p10_px=18.0, p50_px=42.0, confidence=ProbeConfidence.HIGH),
        congestion=Congestion(
            detections_per_frame_mean=6.2,
            mean_pairwise_iou=0.08,
            occlusion_rate=0.15,
            confidence=ProbeConfidence.MEDIUM,
        ),
        motion_dynamics=MotionDynamics(
            optical_flow_mag_p50=1.4,
            per_track_displacement_variance=0.9,
            camera_motion_estimate=0.02,
            confidence=ProbeConfidence.HIGH,
        ),
        lighting=Lighting(
            intensity_variance_over_time=310.5,
            saturation_clipping_fraction=0.01,
            confidence=ProbeConfidence.HIGH,
        ),
        appearance_separability=AppearanceSeparability(
            inter_instance_distance_p50=0.62, confidence=ProbeConfidence.MEDIUM
        ),
        target_novelty=TargetNovelty(
            open_vocab_score_p10=0.31, open_vocab_score_p50=0.55, confidence=ProbeConfidence.LOW
        ),
        catalog_version="v0-2026-09-17",
    )


def _provenance() -> Provenance:
    return Provenance(
        model_id="claude-opus-4-7",
        model_version="4.7.0",
        seed=42,
        catalog_version="v0-2026-09-17",
    )


def _cost_est() -> CostMeasurement:
    return CostMeasurement(
        config={"input_res": 640, "tile_count": 4, "confidence_threshold": 0.4},
        latency_ms=82.0,
        vram_mb=1800.0,
    )


def _cost_measured() -> TargetCostMeasurement:
    return TargetCostMeasurement(
        component_name="general_closed_set_v1",
        catalog_version="v0-2026-09-17",
        hardware=HardwareEnvelope(gpu_model="NVIDIA L4 24GB", vram_gb=24, cpu_cores=16),
        config={"input_res": 640, "tile_count": 4, "confidence_threshold": 0.4},
        latency_ms=210.0,
        vram_mb=3400.0,
        measured_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
    )


def _single_variable_entry(**overrides) -> LedgerEntry:
    data = dict(
        id="entry-0001",
        created_at=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
        parent_entry_id=None,
        task_contract_name="dock_surveillance_v0",
        full_config={
            "topology.detector": "general_closed_set_v1",
            "topology.tile": "on",
            "topology.tracker": "motion_only_v1",
            "topology.reid": "off",
            "detector.confidence_threshold": 0.4,
            "detector.input_res": 640,
            "tile.tile_count": 4,
        },
        changed_parameters=("detector.confidence_threshold",),
        scene_properties=_scene(),
        hypothesis=(
            Prediction(
                metric_name="short_track_ratio",
                baseline_value=0.42,
                predicted_delta=-0.10,
                tolerance=0.03,
            ),
        ),
        evidence=(MetricReading(name="short_track_ratio", value=0.42),),
        outcome=(MetricReading(name="short_track_ratio", value=0.33),),
        verdict=Verdict.CONFIRMED,
        scorer_tier=ScorerTier.LABEL_FREE_PROXY,
        confounded=False,
        cost_estimated=_cost_est(),
        cost_measured=None,
        provenance=_provenance(),
        narrative=(
            "Raised confidence threshold from 0.30 to 0.40 to prune spurious "
            "detections at the loading-dock periphery. short_track_ratio fell "
            "from 0.42 to 0.33 as predicted."
        ),
    )
    data.update(overrides)
    return LedgerEntry(**data)


# --- happy path -------------------------------------------------------------


def test_valid_entry_constructs():
    e = _single_variable_entry()
    assert e.verdict is Verdict.CONFIRMED
    assert e.scorer_tier is ScorerTier.LABEL_FREE_PROXY
    assert e.confounded is False
    assert e.cost_measured is None


def test_entry_with_target_cost_measured():
    e = _single_variable_entry(cost_measured=_cost_measured())
    assert e.cost_measured is not None
    assert e.cost_measured.hardware.gpu_model == "NVIDIA L4 24GB"


def test_multi_variable_entry_requires_confounded_true():
    # §9: multi-variable changes are always confounded.
    e = _single_variable_entry(
        changed_parameters=("detector.confidence_threshold", "tile.tile_count"),
        confounded=True,
    )
    assert e.confounded is True


def test_tier3_reading_with_per_stratum_ci():
    reading = MetricReading(
        name="coco_map_v1",
        value=0.78,
        stratum="low_pixels_high_congestion",
        ci_low=0.72,
        ci_high=0.83,
    )
    assert reading.ci_low == 0.72


# --- §9 invariants ---------------------------------------------------------


def test_multi_variable_without_confounded_rejected():
    with pytest.raises(ValueError):
        _single_variable_entry(
            changed_parameters=("detector.confidence_threshold", "tile.tile_count"),
            confounded=False,
        )


def test_joint_pass_single_variable_may_be_confounded():
    # confounded=True is allowed with a single-variable change (joint-refinement pass).
    e = _single_variable_entry(confounded=True)
    assert e.confounded is True


def test_changed_parameter_must_appear_in_full_config():
    with pytest.raises(ValueError):
        _single_variable_entry(changed_parameters=("detector.does_not_exist",))


def test_changed_parameters_duplicates_rejected():
    with pytest.raises(ValueError):
        _single_variable_entry(
            changed_parameters=("detector.confidence_threshold", "detector.confidence_threshold"),
        )


def test_self_parent_rejected():
    with pytest.raises(ValueError):
        _single_variable_entry(parent_entry_id="entry-0001")  # same as id


def test_empty_hypothesis_rejected():
    with pytest.raises(ValidationError):
        _single_variable_entry(hypothesis=())


def test_empty_outcome_rejected():
    with pytest.raises(ValidationError):
        _single_variable_entry(outcome=())


def test_empty_evidence_rejected():
    with pytest.raises(ValidationError):
        _single_variable_entry(evidence=())


# --- prediction invariants --------------------------------------------------


def test_prediction_tolerance_must_be_positive():
    with pytest.raises(ValidationError):
        Prediction(
            metric_name="x",
            baseline_value=0.5,
            predicted_delta=-0.1,
            tolerance=0.0,
        )


# --- metric-reading invariants ----------------------------------------------


def test_metric_reading_ci_low_above_high_rejected():
    with pytest.raises(ValueError):
        MetricReading(name="m", value=0.5, ci_low=0.9, ci_high=0.7)


def test_metric_reading_bare_scalar_valid():
    # Tier-1 readings have no CI.
    r = MetricReading(name="short_track_ratio", value=0.33)
    assert r.stratum is None and r.ci_low is None


# --- provenance --------------------------------------------------------------


def test_provenance_fields_required():
    with pytest.raises(ValidationError):
        Provenance(model_id="", model_version="1", seed=1, catalog_version="v0")


# --- schema-drift guards ----------------------------------------------------


def test_frozen():
    e = _single_variable_entry()
    with pytest.raises(ValidationError):
        e.verdict = Verdict.REFUTED  # type: ignore[misc]


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        LedgerEntry.model_validate(
            _single_variable_entry().model_dump() | {"undocumented": 1}
        )


def test_verdict_enum_only():
    with pytest.raises(ValidationError):
        LedgerEntry.model_validate(
            _single_variable_entry().model_dump() | {"verdict": "maybe"}
        )


def test_scorer_tier_enum_only():
    with pytest.raises(ValidationError):
        LedgerEntry.model_validate(
            _single_variable_entry().model_dump() | {"scorer_tier": 99}
        )
