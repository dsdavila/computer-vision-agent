"""RegimeVector — §4 profiler output, §4.1 per-axis confidence.

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


class PixelsOnTargetReading(BaseModel):
    """§4 — proposal box height distribution. Proposal-dependent (§4.1)."""

    model_config = _reading

    p10_px: float = Field(ge=0)
    p50_px: float = Field(ge=0)
    confidence: ProbeConfidence


class CongestionReading(BaseModel):
    """§4 — detection density and occlusion. Proposal-dependent (§4.1)."""

    model_config = _reading

    detections_per_frame_mean: float = Field(ge=0)
    mean_pairwise_iou: float = Field(ge=0, le=1)
    occlusion_rate: float = Field(ge=0, le=1)
    confidence: ProbeConfidence


class MotionDynamicsReading(BaseModel):
    """§4 — flow magnitude, displacement variance, camera motion. Detector-independent."""

    model_config = _reading

    optical_flow_mag_p50: float = Field(ge=0)
    per_track_displacement_variance: float = Field(ge=0)
    camera_motion_estimate: float = Field(ge=0)
    confidence: ProbeConfidence


class LightingReading(BaseModel):
    """§4 — intensity variance and clipping. Detector-independent."""

    model_config = _reading

    intensity_variance_over_time: float = Field(ge=0)
    saturation_clipping_fraction: float = Field(ge=0, le=1)
    confidence: ProbeConfidence


class AppearanceSeparabilityReading(BaseModel):
    """§4 — inter-instance embedding distance. Proposal-dependent (§4.1)."""

    model_config = _reading

    inter_instance_distance_p50: float = Field(ge=0)
    confidence: ProbeConfidence


class TargetNoveltyReading(BaseModel):
    """§4 — open-vocab score distribution against the spec ontology. Proposal-dependent (§4.1)."""

    model_config = _reading

    open_vocab_score_p10: float = Field(ge=0, le=1)
    open_vocab_score_p50: float = Field(ge=0, le=1)
    confidence: ProbeConfidence


class RegimeVector(BaseModel):
    """§4 profiler output — all six axes required.

    Every reading carries its own confidence (§4.1). The catalog version pin lets
    §8 feasibility thresholds and §9 ledger entries reason about drift and
    migrations.
    """

    model_config = _reading

    pixels_on_target: PixelsOnTargetReading
    congestion: CongestionReading
    motion_dynamics: MotionDynamicsReading
    lighting: LightingReading
    appearance_separability: AppearanceSeparabilityReading
    target_novelty: TargetNoveltyReading

    catalog_version: str = Field(min_length=1)
