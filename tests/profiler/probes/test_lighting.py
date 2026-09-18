"""Lighting probe tests. Uses synthetic numpy frames so we can control the
statistics exactly — no video decoding involved at this layer."""

from __future__ import annotations

import numpy as np
import pytest

from cv_agent.profiler.probes import measure_lighting
from cv_agent.profiler.video import Frame
from cv_agent.schemas import Lighting, ProbeConfidence


def _grey_frame(index: int, gray: int, size: tuple[int, int] = (48, 64)) -> Frame:
    """Uniform-grey frame at intensity `gray`."""
    arr = np.full((size[0], size[1], 3), gray, dtype=np.uint8)
    return Frame(array=arr, index=index, timestamp_s=index / 30.0)


def _random_frame(seed: int, size: tuple[int, int] = (48, 64)) -> Frame:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
    return Frame(array=arr, index=seed, timestamp_s=0.0)


# --- output structure -------------------------------------------------------


def test_returns_lighting_reading():
    frames = [_grey_frame(i, 128) for i in range(10)]
    reading = measure_lighting(frames)
    assert isinstance(reading, Lighting)


# --- intensity variance -----------------------------------------------------


def test_variance_zero_for_constant_lighting():
    frames = [_grey_frame(i, 128) for i in range(20)]
    reading = measure_lighting(frames)
    assert reading.intensity_variance_over_time == pytest.approx(0.0)


def test_variance_positive_for_ramping_lighting():
    # Grey ramping 0..255 across 30 frames — mean intensity moves monotonically.
    frames = [_grey_frame(i, int(round(i * 255 / 29))) for i in range(30)]
    reading = measure_lighting(frames)
    assert reading.intensity_variance_over_time > 100.0


def test_variance_larger_for_bigger_swings():
    # Two-frame binary flicker between 0 and 255 → largest possible variance
    # for a two-point sample.
    flicker = [_grey_frame(i, 0 if i % 2 == 0 else 255) for i in range(20)]
    smooth = [_grey_frame(i, int(round(i * 255 / 19))) for i in range(20)]
    assert (
        measure_lighting(flicker).intensity_variance_over_time
        > measure_lighting(smooth).intensity_variance_over_time
    )


# --- clipping fraction ------------------------------------------------------


def test_clipping_fraction_zero_for_mid_grey():
    frames = [_grey_frame(i, 128) for i in range(10)]
    reading = measure_lighting(frames)
    assert reading.saturation_clipping_fraction == pytest.approx(0.0)


def test_clipping_fraction_one_for_all_white():
    frames = [_grey_frame(i, 255) for i in range(10)]
    reading = measure_lighting(frames)
    assert reading.saturation_clipping_fraction == pytest.approx(1.0)


def test_clipping_fraction_one_for_all_black():
    frames = [_grey_frame(i, 0) for i in range(10)]
    reading = measure_lighting(frames)
    assert reading.saturation_clipping_fraction == pytest.approx(1.0)


def test_clipping_fraction_bounded_in_unit_interval():
    frames = [_random_frame(i) for i in range(5)]
    reading = measure_lighting(frames)
    assert 0.0 <= reading.saturation_clipping_fraction <= 1.0


# --- confidence tiers -------------------------------------------------------


def test_confidence_high_for_large_sample():
    frames = [_grey_frame(i, 128) for i in range(20)]
    assert measure_lighting(frames).confidence is ProbeConfidence.HIGH


def test_confidence_medium_for_medium_sample():
    frames = [_grey_frame(i, 128) for i in range(10)]
    assert measure_lighting(frames).confidence is ProbeConfidence.MEDIUM


def test_confidence_low_for_small_sample():
    frames = [_grey_frame(i, 128) for i in range(3)]
    assert measure_lighting(frames).confidence is ProbeConfidence.LOW


# --- error paths ------------------------------------------------------------


def test_empty_frames_rejected():
    with pytest.raises(ValueError):
        measure_lighting([])
