"""Runtime-types tests for Box, Proposal, FrameProposals, and the
ProposalDetector Protocol."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from cv_agent.profiler.proposals import Box, FrameProposals, Proposal, ProposalDetector
from cv_agent.profiler.video import Frame


# --- Box --------------------------------------------------------------------


def test_box_basic_geometry():
    b = Box(x1=10, y1=20, x2=30, y2=50)
    assert b.width == 20
    assert b.height == 30
    assert b.area == 600
    assert b.cx == 20
    assert b.cy == 35


def test_box_rejects_reversed_corners():
    with pytest.raises(ValueError):
        Box(x1=30, y1=0, x2=10, y2=20)
    with pytest.raises(ValueError):
        Box(x1=0, y1=20, x2=10, y2=10)


def test_box_rejects_degenerate_zero_width_or_height():
    with pytest.raises(ValueError):
        Box(x1=10, y1=10, x2=10, y2=20)
    with pytest.raises(ValueError):
        Box(x1=10, y1=10, x2=20, y2=10)


def test_box_is_frozen():
    b = Box(x1=0, y1=0, x2=1, y2=1)
    with pytest.raises(Exception):  # dataclass FrozenInstanceError
        b.x1 = 5  # type: ignore[misc]


# --- IoU --------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    b = Box(x1=0, y1=0, x2=10, y2=10)
    assert b.iou(b) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = Box(x1=0, y1=0, x2=10, y2=10)
    b = Box(x1=20, y1=20, x2=30, y2=30)
    assert a.iou(b) == 0.0


def test_iou_touching_boxes_is_zero():
    # Boxes sharing an edge only.
    a = Box(x1=0, y1=0, x2=10, y2=10)
    b = Box(x1=10, y1=0, x2=20, y2=10)
    assert a.iou(b) == 0.0


def test_iou_half_overlap():
    a = Box(x1=0, y1=0, x2=10, y2=10)
    b = Box(x1=5, y1=0, x2=15, y2=10)
    # intersection area = 50; union = 150; iou = 1/3.
    assert a.iou(b) == pytest.approx(1 / 3)


def test_iou_symmetric():
    a = Box(x1=0, y1=0, x2=10, y2=10)
    b = Box(x1=3, y1=4, x2=13, y2=14)
    assert a.iou(b) == pytest.approx(b.iou(a))


# --- Proposal ---------------------------------------------------------------


def test_proposal_valid():
    b = Box(x1=0, y1=0, x2=10, y2=10)
    p = Proposal(box=b, label="person", score=0.9)
    assert p.label == "person"
    assert p.embedding is None


def test_proposal_with_embedding():
    b = Box(x1=0, y1=0, x2=10, y2=10)
    emb = np.zeros(128, dtype=np.float32)
    p = Proposal(box=b, label="person", score=0.9, embedding=emb)
    assert p.embedding is emb


def test_proposal_score_out_of_range_rejected():
    b = Box(x1=0, y1=0, x2=10, y2=10)
    with pytest.raises(ValueError):
        Proposal(box=b, label="person", score=1.1)
    with pytest.raises(ValueError):
        Proposal(box=b, label="person", score=-0.01)


def test_proposal_empty_label_rejected():
    b = Box(x1=0, y1=0, x2=10, y2=10)
    with pytest.raises(ValueError):
        Proposal(box=b, label="", score=0.5)


def test_proposal_is_frozen():
    b = Box(x1=0, y1=0, x2=10, y2=10)
    p = Proposal(box=b, label="person", score=0.5)
    with pytest.raises(Exception):
        p.score = 0.9  # type: ignore[misc]


# --- FrameProposals ---------------------------------------------------------


def test_frame_proposals_defaults_to_empty_tuple():
    frame = Frame(array=np.zeros((10, 10, 3), dtype=np.uint8), index=0, timestamp_s=0.0)
    fp = FrameProposals(frame=frame)
    assert fp.proposals == ()


def test_frame_proposals_carries_multiple():
    frame = Frame(array=np.zeros((10, 10, 3), dtype=np.uint8), index=0, timestamp_s=0.0)
    b1 = Box(x1=0, y1=0, x2=5, y2=5)
    b2 = Box(x1=5, y1=5, x2=10, y2=10)
    fp = FrameProposals(
        frame=frame,
        proposals=(
            Proposal(box=b1, label="person", score=0.9),
            Proposal(box=b2, label="vehicle", score=0.7),
        ),
    )
    assert len(fp.proposals) == 2


# --- ProposalDetector Protocol ---------------------------------------------


class _StubDetector:
    """Minimal fake detector — returns configured proposals per frame index.
    Lives here rather than production code because it exists only to
    demonstrate Protocol compliance in tests."""

    def __init__(self, per_index: dict[int, tuple[Proposal, ...]]):
        self._per_index = per_index

    def propose(
        self,
        frames: Sequence[Frame],
        targets: Sequence[str],
    ) -> Sequence[FrameProposals]:
        del targets  # stub ignores targets
        return [
            FrameProposals(frame=f, proposals=self._per_index.get(f.index, ()))
            for f in frames
        ]


def test_stub_detector_matches_protocol():
    # Structural typing: a class that implements `propose` satisfies the
    # ProposalDetector Protocol without inheritance.
    detector: ProposalDetector = _StubDetector({})
    frame = Frame(array=np.zeros((10, 10, 3), dtype=np.uint8), index=0, timestamp_s=0.0)
    out = detector.propose([frame], targets=["person"])
    assert len(out) == 1
    assert out[0].frame is frame
    assert out[0].proposals == ()


def test_stub_detector_returns_per_frame_proposals():
    b = Box(x1=0, y1=0, x2=5, y2=5)
    person = Proposal(box=b, label="person", score=0.8)
    detector: ProposalDetector = _StubDetector(per_index={0: (person,), 1: ()})

    frames = [
        Frame(array=np.zeros((10, 10, 3), dtype=np.uint8), index=0, timestamp_s=0.0),
        Frame(array=np.zeros((10, 10, 3), dtype=np.uint8), index=1, timestamp_s=0.03),
    ]
    out = detector.propose(frames, targets=["person"])
    assert len(out) == 2
    assert len(out[0].proposals) == 1
    assert out[1].proposals == ()


def test_protocol_accepts_bare_callable_with_matching_signature():
    """Any object with a compatible `propose` signature is a ProposalDetector,
    no ABC inheritance required."""

    def _propose(
        frames: Sequence[Frame], targets: Sequence[str]
    ) -> Sequence[FrameProposals]:
        return [FrameProposals(frame=f) for f in frames]

    class _Wrapper:
        propose = staticmethod(_propose)

    detector: ProposalDetector = _Wrapper()
    frame = Frame(array=np.zeros((5, 5, 3), dtype=np.uint8), index=0, timestamp_s=0.0)
    out = detector.propose([frame], targets=[])
    assert out[0].proposals == ()
