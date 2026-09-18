"""Motion dynamics probe tests. Uses synthetic textured frames (a random
blob pattern re-generated deterministically) so LK tracking has real corners
to lock onto, and the ground-truth motion is known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from cv_agent.profiler.probes import measure_motion_dynamics
from cv_agent.profiler.video import Frame
from cv_agent.schemas import MotionDynamics, ProbeConfidence


H, W = 96, 128


def _textured_frame(seed: int, index: int) -> np.ndarray:
    """A high-contrast blob pattern. Random rectangles on a grey background —
    gives LK enough corners to lock onto for stable tracking."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, W - 12)), int(rng.integers(0, H - 12))
        w, h = int(rng.integers(4, 12)), int(rng.integers(4, 12))
        color = int(rng.integers(0, 256))
        img[y : y + h, x : x + w] = color
    return img


def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate an image by (dx, dy). Pixels revealed at the edges wrap
    around from the opposite side (cheap edge treatment; LK is robust to it
    because most of the interior moves consistently)."""
    return np.roll(img, shift=(dy, dx), axis=(0, 1))


def _make_pair(base: np.ndarray, dx: int, dy: int, index: int) -> tuple[Frame, Frame]:
    a = Frame(array=base, index=index, timestamp_s=index / 30.0)
    b_arr = _shift(base, dx, dy)
    b = Frame(array=b_arr, index=index + 1, timestamp_s=(index + 1) / 30.0)
    return a, b


# --- output structure -------------------------------------------------------


def test_returns_motion_dynamics_reading():
    base = _textured_frame(0, 0)
    pairs = [_make_pair(base, dx=0, dy=0, index=i) for i in range(10)]
    reading = measure_motion_dynamics(pairs)
    assert isinstance(reading, MotionDynamics)


# --- static scene -----------------------------------------------------------


def test_static_scene_returns_near_zero():
    base = _textured_frame(0, 0)
    pairs = [_make_pair(base, dx=0, dy=0, index=i) for i in range(10)]
    reading = measure_motion_dynamics(pairs)
    assert reading.optical_flow_mag_p50 < 1.0
    assert reading.camera_motion_estimate < 1.0
    assert reading.per_track_displacement_variance < 1.0


# --- uniform camera pan -----------------------------------------------------


def test_uniform_pan_recovers_camera_motion():
    # Every pair: shift 5 pixels right. Every point should move ~5 to the
    # right; median vector magnitude ≈ 5; residuals ≈ 0.
    base = _textured_frame(1, 0)
    pairs = [_make_pair(base, dx=5, dy=0, index=i) for i in range(10)]
    reading = measure_motion_dynamics(pairs)

    assert reading.camera_motion_estimate == pytest.approx(5.0, abs=1.0)
    assert reading.optical_flow_mag_p50 == pytest.approx(5.0, abs=1.0)
    # Bulk motion → residuals should be small.
    assert reading.per_track_displacement_variance < 1.0


# --- mixed motion -----------------------------------------------------------


def test_mixed_motion_produces_positive_residual_variance():
    """Half the frame pairs pan right, half pan up. Camera motion averages
    to something small; per-pair residuals stay small (each pair is uniform)
    but that isn't what "diverging objects" would look like. To synthesize
    genuinely diverging motion inside a single pair, we blend two shifted
    copies: a translated background plus a foreground blob moving the
    opposite way."""
    rng = np.random.default_rng(2)
    base = _textured_frame(2, 0)
    pairs: list[tuple[Frame, Frame]] = []
    for i in range(10):
        # Background pans right by 5.
        bg = _shift(base, dx=5, dy=0)
        # Foreground blob at a known location moves left by 5 relative to the
        # background pan (so absolute shift = 0 for the blob region).
        blob_x = int(rng.integers(20, W - 40))
        blob_y = int(rng.integers(20, H - 40))
        bg[blob_y : blob_y + 24, blob_x : blob_x + 24] = base[
            blob_y : blob_y + 24, blob_x : blob_x + 24
        ]
        a = Frame(array=base, index=i, timestamp_s=i / 30.0)
        b = Frame(array=bg, index=i + 1, timestamp_s=(i + 1) / 30.0)
        pairs.append((a, b))

    reading = measure_motion_dynamics(pairs)
    # There's real motion in the scene.
    assert reading.optical_flow_mag_p50 > 1.0
    # Objects diverge from the bulk pan → residual variance is meaningfully
    # nonzero. Compare against a static-scene control to keep the assertion
    # robust to sub-pixel LK jitter.
    static_pairs = [_make_pair(base, dx=0, dy=0, index=i) for i in range(10)]
    static_reading = measure_motion_dynamics(static_pairs)
    assert reading.per_track_displacement_variance > 10 * (
        static_reading.per_track_displacement_variance + 0.05
    )


# --- degenerate input -------------------------------------------------------


def test_untextured_video_returns_low_confidence_zeros():
    # Blank grey frames give goodFeaturesToTrack nothing to lock onto.
    blank = np.full((H, W, 3), 128, dtype=np.uint8)
    pairs = [
        (
            Frame(array=blank, index=i, timestamp_s=i / 30.0),
            Frame(array=blank, index=i + 1, timestamp_s=(i + 1) / 30.0),
        )
        for i in range(10)
    ]
    reading = measure_motion_dynamics(pairs)
    assert reading.confidence is ProbeConfidence.LOW
    assert reading.optical_flow_mag_p50 == 0.0
    assert reading.per_track_displacement_variance == 0.0
    assert reading.camera_motion_estimate == 0.0


def test_mismatched_frame_shapes_rejected():
    a_arr = np.zeros((48, 64, 3), dtype=np.uint8)
    b_arr = np.zeros((96, 128, 3), dtype=np.uint8)
    a = Frame(array=a_arr, index=0, timestamp_s=0.0)
    b = Frame(array=b_arr, index=1, timestamp_s=0.03)
    with pytest.raises(ValueError):
        measure_motion_dynamics([(a, b)])


def test_empty_pairs_rejected():
    with pytest.raises(ValueError):
        measure_motion_dynamics([])


# --- confidence tiers -------------------------------------------------------


def test_confidence_high_for_large_pair_count():
    base = _textured_frame(3, 0)
    pairs = [_make_pair(base, dx=1, dy=0, index=i) for i in range(20)]
    assert measure_motion_dynamics(pairs).confidence is ProbeConfidence.HIGH


def test_confidence_medium_for_medium_pair_count():
    base = _textured_frame(4, 0)
    pairs = [_make_pair(base, dx=1, dy=0, index=i) for i in range(10)]
    assert measure_motion_dynamics(pairs).confidence is ProbeConfidence.MEDIUM


def test_confidence_low_for_few_pairs():
    base = _textured_frame(5, 0)
    pairs = [_make_pair(base, dx=1, dy=0, index=i) for i in range(3)]
    assert measure_motion_dynamics(pairs).confidence is ProbeConfidence.LOW
