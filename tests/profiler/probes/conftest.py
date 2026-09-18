"""Shared fixtures for proposal-dependent probe tests. Builds FrameProposals
from synthetic boxes/scores/embeddings — no video decoding at this layer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cv_agent.profiler.proposals import Box, FrameProposals, Proposal
from cv_agent.profiler.video import Frame


def make_frame(index: int = 0, size: tuple[int, int] = (480, 640)) -> Frame:
    """Return a black Frame at the given index; frames are not read by the
    proposal-dependent probes, which look at proposals only."""
    return Frame(
        array=np.zeros((size[0], size[1], 3), dtype=np.uint8),
        index=index,
        timestamp_s=index / 30.0,
    )


def make_proposal(
    x1: float = 100,
    y1: float = 100,
    x2: float = 140,
    y2: float = 200,
    label: str = "person",
    score: float = 0.9,
    embedding: np.ndarray | None = None,
) -> Proposal:
    return Proposal(
        box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
        label=label,
        score=score,
        embedding=embedding,
    )


def frames_with(
    proposals_per_frame: Sequence[Sequence[Proposal]],
) -> list[FrameProposals]:
    return [
        FrameProposals(frame=make_frame(i), proposals=tuple(props))
        for i, props in enumerate(proposals_per_frame)
    ]
