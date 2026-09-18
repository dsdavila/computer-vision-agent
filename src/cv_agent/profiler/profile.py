"""Profiler composition — the `profile` entry point (§4, §17 M2).

Runs the six §4 probes against a video + a detector, assembles the results
into a SceneProperties, and pins the catalog_version from the TaskContract.

Frame sampling policy:
- Uniform sample of `n_frames` frames for the "spatial" probes (lighting,
  pixels-on-target, congestion, appearance separability, target novelty).
- Consecutive pair sample of `n_motion_pairs` pairs for the motion probe.
Both samplers are deterministic per §4.

The detector is injected — any object matching the ProposalDetector Protocol
works. The profiler does not know or care which implementation. This is the
plug-in point the design's "existing VLM tuner drops in" language (§3, §17
M3) refers to.
"""

from __future__ import annotations

from pathlib import Path

from cv_agent.profiler.probes import (
    measure_appearance_separability,
    measure_congestion,
    measure_lighting,
    measure_motion_dynamics,
    measure_pixels_on_target,
    measure_target_novelty,
)
from cv_agent.profiler.proposals import ProposalDetector
from cv_agent.profiler.video import VideoReader, sample_frame_pairs
from cv_agent.schemas import (
    MotionDynamics,
    ProbeConfidence,
    SceneProperties,
    TaskContract,
)


_DEFAULT_FRAMES = 30
_DEFAULT_MOTION_PAIRS = 20


def profile(
    video: VideoReader | str | Path,
    task_contract: TaskContract,
    detector: ProposalDetector,
    *,
    n_frames: int = _DEFAULT_FRAMES,
    n_motion_pairs: int = _DEFAULT_MOTION_PAIRS,
) -> SceneProperties:
    """Profile a video into SceneProperties.

    All six §4 probes run; the resulting SceneProperties carries the catalog
    version pinned from `task_contract`. The detector-independent probes
    (lighting, motion) run first and don't depend on detector output — a
    detector that returns nothing still yields a valid SceneProperties, with
    proposal-dependent probes reporting LOW confidence per §4.1."""

    reader = video if isinstance(video, VideoReader) else VideoReader(video)

    frames = reader.sample_uniform(n_frames)
    if not frames:
        raise ValueError(f"profile: {reader.path} has no frames to sample")

    motion_pairs = sample_frame_pairs(reader, n_motion_pairs)

    # Detector-independent probes.
    lighting = measure_lighting(frames)
    if motion_pairs:
        motion = measure_motion_dynamics(motion_pairs)
    else:
        motion = _low_confidence_motion()

    # Single detector pass; the four proposal-dependent probes share it.
    targets = task_contract.ontology.all_target_names
    frame_props = list(detector.propose(frames, targets))

    return SceneProperties(
        pixels_on_target=measure_pixels_on_target(frame_props),
        congestion=measure_congestion(frame_props),
        motion_dynamics=motion,
        lighting=lighting,
        appearance_separability=measure_appearance_separability(frame_props),
        target_novelty=measure_target_novelty(frame_props, targets),
        catalog_version=task_contract.catalog_version,
    )


def _low_confidence_motion() -> MotionDynamics:
    """Fallback when the video is too short to form any motion pairs.
    Matches the "no trackable points" branch inside measure_motion_dynamics —
    zeros at LOW confidence, so §8 can refuse on low-confidence profiling."""
    return MotionDynamics(
        optical_flow_mag_p50=0.0,
        per_track_displacement_variance=0.0,
        camera_motion_estimate=0.0,
        confidence=ProbeConfidence.LOW,
    )
