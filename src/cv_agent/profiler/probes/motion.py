"""Motion dynamics probe — §4 detector-independent property (§4.1).

Emits three genuinely distinct numbers about scene motion:

- `optical_flow_mag_p50`: median displacement magnitude across all tracked
  points. How much things move.
- `camera_motion_estimate`: magnitude of the median displacement *vector*
  (median dx, median dy). Bulk translation — mostly camera motion.
- `per_track_displacement_variance`: variance of per-point displacement
  magnitudes after subtracting the per-pair median vector. How much objects
  diverge from the bulk motion.

Chicken-and-egg avoidance: §4 names this "per-track displacement variance",
but the profiler runs before the planner picks a tracker. We proxy with
sparse Lucas-Kanade on Shi–Tomasi corners — the same computation that gives
us flow magnitude and camera motion. Downstream the real tracker replaces
this proxy at inference time; for scene classification it is fit for
purpose.

Detector-independent per §4.1: no bounding boxes or classes involved.
Confidence keys on sample-size tiers (same policy as Lighting), with a floor
override: if not enough points are trackable in any pair to compute a signal,
we return zeros at LOW confidence rather than raise. That lets §8 refuse on
low profiling confidence instead of the pipeline crashing on untextured
video (blank walls, extreme motion blur).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from cv_agent.profiler.video import Frame
from cv_agent.schemas import MotionDynamics, ProbeConfidence


_HIGH_MIN_PAIRS = 20
_MEDIUM_MIN_PAIRS = 8

# Sparse-tracking parameters. Modest defaults; the profiler is deterministic
# and these do not change per video.
_MAX_CORNERS = 200
_QUALITY_LEVEL = 0.01
_MIN_DISTANCE = 8
_MIN_TRACKED_POINTS_PER_PAIR = 5


def measure_motion_dynamics(
    frame_pairs: Sequence[tuple[Frame, Frame]],
) -> MotionDynamics:
    """Detector-independent motion probe.

    Each frame pair is one anchor frame plus its immediate successor. The
    sampler in `profiler.video.sample_frame_pairs` picks anchors evenly
    across the video."""
    if not frame_pairs:
        raise ValueError("measure_motion_dynamics: frame_pairs must be non-empty")

    all_magnitudes: list[np.ndarray] = []
    per_pair_median_vec: list[np.ndarray] = []
    residual_magnitudes: list[np.ndarray] = []

    for a, b in frame_pairs:
        if a.array.shape != b.array.shape:
            raise ValueError(
                "measure_motion_dynamics: frame pair has mismatched shapes "
                f"({a.array.shape} vs {b.array.shape})"
            )
        pts_a = cv2.goodFeaturesToTrack(
            cv2.cvtColor(a.array, cv2.COLOR_BGR2GRAY),
            maxCorners=_MAX_CORNERS,
            qualityLevel=_QUALITY_LEVEL,
            minDistance=_MIN_DISTANCE,
        )
        if pts_a is None or len(pts_a) < _MIN_TRACKED_POINTS_PER_PAIR:
            continue
        pts_b, status, _err = cv2.calcOpticalFlowPyrLK(
            cv2.cvtColor(a.array, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(b.array, cv2.COLOR_BGR2GRAY),
            pts_a,
            None,
        )
        if pts_b is None:
            continue
        good = status.flatten() == 1
        if int(good.sum()) < _MIN_TRACKED_POINTS_PER_PAIR:
            continue
        a_pts = pts_a[good].reshape(-1, 2)
        b_pts = pts_b[good].reshape(-1, 2)
        disp = b_pts - a_pts  # (n, 2)
        mag = np.linalg.norm(disp, axis=1)

        median_vec = np.median(disp, axis=0)  # (2,)
        residual = disp - median_vec
        residual_mag = np.linalg.norm(residual, axis=1)

        all_magnitudes.append(mag)
        per_pair_median_vec.append(median_vec)
        residual_magnitudes.append(residual_mag)

    if not all_magnitudes:
        # No pair produced enough trackable points; report LOW confidence and
        # zeros so §8 can refuse rather than have the pipeline crash.
        return MotionDynamics(
            optical_flow_mag_p50=0.0,
            per_track_displacement_variance=0.0,
            camera_motion_estimate=0.0,
            confidence=ProbeConfidence.LOW,
        )

    flat_mag = np.concatenate(all_magnitudes)
    flat_residual = np.concatenate(residual_magnitudes)
    median_vec_across_pairs = np.median(np.stack(per_pair_median_vec), axis=0)

    return MotionDynamics(
        optical_flow_mag_p50=float(np.median(flat_mag)),
        per_track_displacement_variance=float(np.var(flat_residual)),
        camera_motion_estimate=float(np.linalg.norm(median_vec_across_pairs)),
        confidence=_confidence_from_pair_count(len(frame_pairs)),
    )


def _confidence_from_pair_count(n: int) -> ProbeConfidence:
    if n >= _HIGH_MIN_PAIRS:
        return ProbeConfidence.HIGH
    if n >= _MEDIUM_MIN_PAIRS:
        return ProbeConfidence.MEDIUM
    return ProbeConfidence.LOW
