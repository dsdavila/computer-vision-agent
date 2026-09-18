"""SceneProperties schema tests — validate the frozen-schema invariants.

M1 is schemas only. These tests exercise the shape, not any profiler logic.
"""

import pytest
from pydantic import ValidationError

from cv_agent.schemas import (
    AppearanceSeparability,
    Congestion,
    Lighting,
    MotionDynamics,
    PixelsOnTarget,
    ProbeConfidence,
    SceneProperties,
    TargetNovelty,
)


def _valid_scene() -> SceneProperties:
    return SceneProperties(
        pixels_on_target=PixelsOnTarget(
            p10_px=18.0, p50_px=42.0, confidence=ProbeConfidence.HIGH
        ),
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
            open_vocab_score_p10=0.31,
            open_vocab_score_p50=0.55,
            confidence=ProbeConfidence.LOW,
        ),
        catalog_version="v0-2026-09-17",
    )


def test_valid_scene_constructs():
    s = _valid_scene()
    assert s.pixels_on_target.p50_px == 42.0
    assert s.target_novelty.confidence is ProbeConfidence.LOW
    assert s.catalog_version == "v0-2026-09-17"


def test_missing_confidence_fails():
    # §16 #6 — confidence added later means every historical entry is unlabelled.
    # Missing confidence must be a validation error, not a silent default.
    with pytest.raises(ValidationError):
        PixelsOnTarget(p10_px=18.0, p50_px=42.0)  # type: ignore[call-arg]


def test_missing_property_fails():
    with pytest.raises(ValidationError):
        SceneProperties(  # type: ignore[call-arg]
            pixels_on_target=PixelsOnTarget(
                p10_px=18.0, p50_px=42.0, confidence=ProbeConfidence.HIGH
            ),
            # missing every other property
            catalog_version="v0",
        )


def test_extra_field_forbidden():
    s = _valid_scene()
    with pytest.raises(ValidationError):
        SceneProperties.model_validate(s.model_dump() | {"unexpected_property": 1.0})


def test_extra_field_on_reading_forbidden():
    with pytest.raises(ValidationError):
        PixelsOnTarget.model_validate(
            {"p10_px": 18.0, "p50_px": 42.0, "confidence": "high", "extra": 1}
        )


def test_frozen():
    s = _valid_scene()
    with pytest.raises(ValidationError):
        s.pixels_on_target.p10_px = 999.0  # type: ignore[misc]


def test_negative_pixels_rejected():
    with pytest.raises(ValidationError):
        PixelsOnTarget(p10_px=-1.0, p50_px=42.0, confidence=ProbeConfidence.HIGH)


def test_rate_bounds_enforced():
    # occlusion_rate must be in [0, 1]
    with pytest.raises(ValidationError):
        Congestion(
            detections_per_frame_mean=6.2,
            mean_pairwise_iou=0.08,
            occlusion_rate=1.5,
            confidence=ProbeConfidence.HIGH,
        )


def test_confidence_enum_only():
    with pytest.raises(ValidationError):
        PixelsOnTarget.model_validate(
            {"p10_px": 18.0, "p50_px": 42.0, "confidence": "maybe"}
        )


def test_catalog_version_required_and_nonempty():
    with pytest.raises(ValidationError):
        SceneProperties.model_validate(
            _valid_scene().model_dump() | {"catalog_version": ""}
        )
