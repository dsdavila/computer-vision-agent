"""SceneProperties — §4 profiler output, §4.1 per-property confidence.

Frozen. Everything downstream (feasibility gate, planner, ledger, mapping rules)
compiles against this shape.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProbeConfidence(str, Enum):
    """§4.1 — the feasibility gate refuses on LOW."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_reading = ConfigDict(frozen=True, extra="forbid")


class PixelsOnTarget(BaseModel):
    """§4 — proposal box height distribution. Proposal-dependent (§4.1)."""

    model_config = _reading

    p10_px: float = Field(ge=0)
    p50_px: float = Field(ge=0)
    confidence: ProbeConfidence


class Congestion(BaseModel):
    """§4 — detection density and occlusion. Proposal-dependent (§4.1)."""

    model_config = _reading

    detections_per_frame_mean: float = Field(ge=0)
    mean_pairwise_iou: float = Field(ge=0, le=1)
    occlusion_rate: float = Field(ge=0, le=1)
    confidence: ProbeConfidence


class MotionDynamics(BaseModel):
    """§4 — flow magnitude, displacement variance, camera motion. Detector-independent."""

    model_config = _reading

    optical_flow_mag_p50: float = Field(ge=0)
    per_track_displacement_variance: float = Field(ge=0)
    camera_motion_estimate: float = Field(ge=0)
    confidence: ProbeConfidence


class Lighting(BaseModel):
    """§4 — intensity variance and clipping. Detector-independent."""

    model_config = _reading

    intensity_variance_over_time: float = Field(ge=0)
    saturation_clipping_fraction: float = Field(ge=0, le=1)
    confidence: ProbeConfidence


class AppearanceSeparability(BaseModel):
    """§4 — inter-instance embedding distance. Proposal-dependent (§4.1)."""

    model_config = _reading

    inter_instance_distance_p50: float = Field(ge=0)
    confidence: ProbeConfidence


class TargetNovelty(BaseModel):
    """§4 — open-vocab score distribution against the spec ontology. Proposal-dependent (§4.1)."""

    model_config = _reading

    open_vocab_score_p10: float = Field(ge=0, le=1)
    open_vocab_score_p50: float = Field(ge=0, le=1)
    confidence: ProbeConfidence


class SceneProperties(BaseModel):
    """§4 profiler output — all six properties required.

    Every property carries its own confidence (§4.1). The catalog version pin lets
    §8 feasibility thresholds and §9 ledger entries reason about drift and
    migrations.
    """

    model_config = _reading

    pixels_on_target: PixelsOnTarget
    congestion: Congestion
    motion_dynamics: MotionDynamics
    lighting: Lighting
    appearance_separability: AppearanceSeparability
    target_novelty: TargetNovelty

    catalog_version: str = Field(min_length=1)
