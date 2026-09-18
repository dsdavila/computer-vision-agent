"""Pixels-on-target probe tests."""

from __future__ import annotations

import pytest

from cv_agent.profiler.probes import measure_pixels_on_target
from cv_agent.schemas import PixelsOnTarget, ProbeConfidence

from .conftest import frames_with, make_proposal


def test_returns_pixels_on_target_reading():
    fps = frames_with([[make_proposal(y1=100, y2=200)]] * 10)
    r = measure_pixels_on_target(fps)
    assert isinstance(r, PixelsOnTarget)


def test_p50_matches_uniform_box_height():
    # Every proposal has height 100 -> both p10 and p50 should be 100.
    fps = frames_with([[make_proposal(y1=0, y2=100)]] * 20)
    r = measure_pixels_on_target(fps)
    assert r.p10_px == pytest.approx(100.0)
    assert r.p50_px == pytest.approx(100.0)


def test_p10_below_p50_for_mixed_heights():
    # Ten boxes: heights 10, 20, 30, ..., 100. p10 near 10, p50 near 55.
    fps = frames_with([
        [make_proposal(y1=0, y2=h)] for h in range(10, 101, 10)
    ])
    r = measure_pixels_on_target(fps)
    assert r.p10_px < r.p50_px
    assert 10 <= r.p10_px <= 20
    assert 45 <= r.p50_px <= 65


def test_low_score_proposals_filtered_out():
    fps = frames_with([
        [make_proposal(y1=0, y2=50, score=0.1)]  # below 0.3 threshold
        for _ in range(20)
    ])
    r = measure_pixels_on_target(fps)
    assert r.p50_px == 0.0
    assert r.confidence is ProbeConfidence.LOW


def test_confidence_tiers():
    # HIGH: 20 frames with 2 boxes each = 40 proposals
    r = measure_pixels_on_target(frames_with([[make_proposal()] * 2] * 20))
    assert r.confidence is ProbeConfidence.HIGH

    # MEDIUM: 10 frames, 1 box each = 10 proposals
    r = measure_pixels_on_target(frames_with([[make_proposal()]] * 10))
    assert r.confidence is ProbeConfidence.MEDIUM

    # LOW: 3 frames, 1 box each
    r = measure_pixels_on_target(frames_with([[make_proposal()]] * 3))
    assert r.confidence is ProbeConfidence.LOW


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        measure_pixels_on_target([])


def test_frames_without_proposals_return_low_confidence():
    # Frames sampled but detector found nothing.
    fps = frames_with([[]] * 20)
    r = measure_pixels_on_target(fps)
    assert r.confidence is ProbeConfidence.LOW
    assert r.p50_px == 0.0
