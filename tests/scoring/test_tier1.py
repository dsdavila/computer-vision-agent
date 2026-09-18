"""Tier-1 scorer tests. Metric correctness + §6 cross-topology-comparison
enforcement."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from cv_agent.profiler.proposals import Box, FrameProposals, Proposal
from cv_agent.profiler.video import Frame
from cv_agent.scoring import (
    CrossTopologyComparisonError,
    PipelineOutput,
    Tier1Score,
    Track,
    TrackedFrame,
    compare_tier1,
    compute_tier1,
)

from .conftest import make_detector, make_topology


# --- helpers ---------------------------------------------------------------


def _frame(index: int) -> Frame:
    return Frame(
        array=np.zeros((10, 10, 3), dtype=np.uint8),
        index=index,
        timestamp_s=index / 30.0,
    )


def _frame_props(index: int, scores: Sequence[float]) -> FrameProposals:
    box = Box(x1=10, y1=10, x2=30, y2=50)
    props = tuple(Proposal(box=box, label="person", score=s) for s in scores)
    return FrameProposals(frame=_frame(index), proposals=props)


def _track(track_id: int, frame_indices: Sequence[int]) -> Track:
    box = Box(x1=10, y1=10, x2=30, y2=50)
    frames = tuple(
        TrackedFrame(frame_index=i, box=box, score=0.9) for i in frame_indices
    )
    return Track(id=track_id, frames=frames)


def _pipeline(
    *,
    detector_threshold: float = 0.4,
    frame_proposals: tuple[FrameProposals, ...] = (),
    tracks: tuple[Track, ...] = (),
) -> PipelineOutput:
    return PipelineOutput(
        frame_proposals=frame_proposals,
        tracks=tracks,
        detector_threshold=detector_threshold,
    )


# --- individual proxies ---------------------------------------------------


def test_short_track_ratio_all_short():
    # 3 short tracks (< 5 frames each) → ratio 1.0.
    tracks = (_track(1, [0, 1]), _track(2, [3, 4, 5]), _track(3, [10]))
    out = _pipeline(
        frame_proposals=(_frame_props(0, [0.9]),),
        tracks=tracks,
    )
    score = compute_tier1(make_topology(), out)
    assert score.short_track_ratio == pytest.approx(1.0)


def test_short_track_ratio_none_short():
    # 2 tracks, both ≥ 5 frames.
    tracks = (_track(1, list(range(10))), _track(2, list(range(5, 15))))
    out = _pipeline(
        frame_proposals=(_frame_props(0, [0.9]),),
        tracks=tracks,
    )
    score = compute_tier1(make_topology(), out)
    assert score.short_track_ratio == pytest.approx(0.0)


def test_short_track_ratio_empty_tracks_returns_zero():
    out = _pipeline(frame_proposals=(_frame_props(0, [0.9]),))
    score = compute_tier1(make_topology(), out)
    assert score.short_track_ratio == 0.0


def test_conf_p50_margin_positive_when_scores_above_threshold():
    fps = tuple(_frame_props(i, [0.8, 0.85, 0.9]) for i in range(10))
    out = _pipeline(detector_threshold=0.4, frame_proposals=fps)
    score = compute_tier1(make_topology(), out)
    # p50 of [0.8, 0.85, 0.9]*10 = 0.85; minus 0.4 = 0.45.
    assert score.conf_p50_margin == pytest.approx(0.45)


def test_conf_p50_margin_zero_when_scores_at_threshold():
    fps = tuple(_frame_props(i, [0.4]) for i in range(5))
    out = _pipeline(detector_threshold=0.4, frame_proposals=fps)
    score = compute_tier1(make_topology(), out)
    assert score.conf_p50_margin == pytest.approx(0.0)


def test_conf_p50_margin_empty_returns_zero():
    out = _pipeline(detector_threshold=0.4)
    score = compute_tier1(make_topology(), out)
    assert score.conf_p50_margin == 0.0


def test_near_threshold_fraction_high_when_scores_at_edge():
    # All scores within 0.1 of threshold 0.4.
    fps = tuple(_frame_props(i, [0.35, 0.42, 0.48]) for i in range(5))
    out = _pipeline(detector_threshold=0.4, frame_proposals=fps)
    score = compute_tier1(make_topology(), out)
    assert score.near_threshold_fraction == pytest.approx(1.0)


def test_near_threshold_fraction_low_when_scores_far_from_edge():
    fps = tuple(_frame_props(i, [0.9, 0.85, 0.8]) for i in range(5))
    out = _pipeline(detector_threshold=0.4, frame_proposals=fps)
    score = compute_tier1(make_topology(), out)
    assert score.near_threshold_fraction == pytest.approx(0.0)


def test_track_count_stability_high_when_constant():
    fps = tuple(_frame_props(i, [0.9]) for i in range(10))
    # Two tracks, both covering all 10 frames → per-frame count constant at 2.
    tracks = (_track(1, list(range(10))), _track(2, list(range(10))))
    out = _pipeline(frame_proposals=fps, tracks=tracks)
    score = compute_tier1(make_topology(), out)
    assert score.track_count_stability == pytest.approx(1.0)


def test_track_count_stability_lower_when_oscillating():
    fps = tuple(_frame_props(i, [0.9]) for i in range(10))
    # Track alive on odd frames only → per-frame counts [0, 1, 0, 1, ...].
    tracks = (_track(1, [1, 3, 5, 7, 9]),)
    out = _pipeline(frame_proposals=fps, tracks=tracks)
    score = compute_tier1(make_topology(), out)
    # cv = 1.0 → stability = 0.0.
    assert score.track_count_stability == pytest.approx(0.0, abs=1e-6)


def test_track_count_stability_empty_returns_one():
    out = _pipeline()
    score = compute_tier1(make_topology(), out)
    assert score.track_count_stability == 1.0


# --- score object -----------------------------------------------------------


def test_score_carries_topology():
    topology = make_topology()
    out = _pipeline(frame_proposals=(_frame_props(0, [0.9]),))
    score = compute_tier1(topology, out)
    assert score.topology is topology


def test_score_is_frozen():
    score = compute_tier1(
        make_topology(),
        _pipeline(frame_proposals=(_frame_props(0, [0.9]),)),
    )
    with pytest.raises(Exception):
        score.short_track_ratio = 0.99  # type: ignore[misc]


# --- cross-topology ordering is refused -----------------------------------


def _small_score(topology) -> Tier1Score:
    return compute_tier1(topology, _pipeline(frame_proposals=(_frame_props(0, [0.9]),)))


def test_lt_raises_with_helpful_message():
    a = _small_score(make_topology())
    b = _small_score(make_topology(detector=make_detector(name="other_detector")))
    with pytest.raises(TypeError, match="not orderable"):
        a < b


def test_gt_raises():
    a = _small_score(make_topology())
    b = _small_score(make_topology(detector=make_detector(name="other")))
    with pytest.raises(TypeError):
        a > b


def test_le_ge_raise():
    a = _small_score(make_topology())
    b = _small_score(make_topology(detector=make_detector(name="other")))
    with pytest.raises(TypeError):
        a <= b
    with pytest.raises(TypeError):
        a >= b


# --- compare_tier1 ---------------------------------------------------------


def test_compare_tier1_same_topology_higher_margin_wins():
    topology = make_topology()
    a = compute_tier1(topology, _pipeline(
        detector_threshold=0.4,
        frame_proposals=(_frame_props(0, [0.9]),),
    ))
    b = compute_tier1(topology, _pipeline(
        detector_threshold=0.4,
        frame_proposals=(_frame_props(0, [0.5]),),
    ))
    assert compare_tier1(a, b, on="conf_p50_margin") == 1
    assert compare_tier1(b, a, on="conf_p50_margin") == -1


def test_compare_tier1_short_track_ratio_lower_wins():
    # short_track_ratio direction: lower is better.
    topology = make_topology()
    a_out = _pipeline(
        frame_proposals=(_frame_props(0, [0.9]),),
        tracks=(_track(1, list(range(10))),),  # long track → ratio 0
    )
    b_out = _pipeline(
        frame_proposals=(_frame_props(0, [0.9]),),
        tracks=(_track(1, [0, 1]),),  # short track → ratio 1
    )
    a = compute_tier1(topology, a_out)
    b = compute_tier1(topology, b_out)
    assert compare_tier1(a, b, on="short_track_ratio") == 1


def test_compare_tier1_equal_returns_zero():
    topology = make_topology()
    out = _pipeline(frame_proposals=(_frame_props(0, [0.9]),))
    a = compute_tier1(topology, out)
    b = compute_tier1(topology, out)
    for metric in ("short_track_ratio", "conf_p50_margin", "near_threshold_fraction", "track_count_stability"):
        assert compare_tier1(a, b, on=metric) == 0


def test_compare_tier1_cross_topology_raises():
    a = _small_score(make_topology())
    b = _small_score(make_topology(detector=make_detector(name="other_detector")))
    with pytest.raises(CrossTopologyComparisonError):
        compare_tier1(a, b, on="conf_p50_margin")


def test_compare_tier1_unknown_metric_raises():
    topology = make_topology()
    a = _small_score(topology)
    b = _small_score(topology)
    with pytest.raises(ValueError, match="unknown metric"):
        compare_tier1(a, b, on="not_a_real_metric")
