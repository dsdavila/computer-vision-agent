"""Congestion probe tests."""

from __future__ import annotations

import pytest

from cv_agent.profiler.probes import measure_congestion
from cv_agent.schemas import Congestion, ProbeConfidence

from .conftest import frames_with, make_proposal


def test_returns_congestion_reading():
    fps = frames_with([[make_proposal()]] * 10)
    r = measure_congestion(fps)
    assert isinstance(r, Congestion)


def test_detections_per_frame_mean():
    fps = frames_with([[make_proposal()] * 5] * 10)  # 5 dets per frame, 10 frames
    r = measure_congestion(fps)
    assert r.detections_per_frame_mean == pytest.approx(5.0)


def test_mean_pairwise_iou_zero_for_disjoint_boxes():
    # Boxes placed side by side, no overlap.
    fps = frames_with([[
        make_proposal(x1=0, y1=0, x2=10, y2=10),
        make_proposal(x1=100, y1=0, x2=110, y2=10),
        make_proposal(x1=200, y1=0, x2=210, y2=10),
    ]] * 10)
    r = measure_congestion(fps)
    assert r.mean_pairwise_iou == pytest.approx(0.0)
    assert r.occlusion_rate == pytest.approx(0.0)


def test_mean_pairwise_iou_positive_for_overlapping_boxes():
    # Two identical boxes → iou = 1, mean pairwise iou = 1.
    fps = frames_with([[
        make_proposal(x1=0, y1=0, x2=10, y2=10),
        make_proposal(x1=0, y1=0, x2=10, y2=10),
    ]] * 10)
    r = measure_congestion(fps)
    assert r.mean_pairwise_iou == pytest.approx(1.0)
    assert r.occlusion_rate == pytest.approx(1.0)


def test_occlusion_rate_partial_overlap():
    # Two boxes 50% overlap (iou = 1/3 > 0.3 threshold → both count as occluded);
    # a third disjoint (not occluded).
    fps = frames_with([[
        make_proposal(x1=0, y1=0, x2=10, y2=10),
        make_proposal(x1=5, y1=0, x2=15, y2=10),
        make_proposal(x1=200, y1=200, x2=210, y2=210),
    ]] * 10)
    r = measure_congestion(fps)
    # 2 of 3 proposals occluded per frame → 2/3.
    assert r.occlusion_rate == pytest.approx(2 / 3)


def test_low_score_proposals_filtered():
    fps = frames_with([[make_proposal(score=0.1)]] * 20)
    r = measure_congestion(fps)
    assert r.detections_per_frame_mean == 0.0
    assert r.confidence is ProbeConfidence.LOW


def test_single_box_frames_have_zero_pairwise_iou():
    fps = frames_with([[make_proposal()]] * 20)
    r = measure_congestion(fps)
    assert r.mean_pairwise_iou == pytest.approx(0.0)
    assert r.occlusion_rate == pytest.approx(0.0)


def test_confidence_tiers():
    r = measure_congestion(frames_with([[make_proposal()] * 2] * 20))
    assert r.confidence is ProbeConfidence.HIGH

    r = measure_congestion(frames_with([[make_proposal()]] * 10))
    assert r.confidence is ProbeConfidence.MEDIUM

    r = measure_congestion(frames_with([[make_proposal()]] * 3))
    assert r.confidence is ProbeConfidence.LOW


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        measure_congestion([])
