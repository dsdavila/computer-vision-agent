"""Pixels-on-target probe — §4 proposal-dependent property.

Emits p10 and p50 of proposal box heights across the sampled frames. Height,
not diagonal, because the design uses box height (§4 table) — a stable
across-view axis when targets stand upright.

Proposal-dependent per §4.1: confidence keys on how many high-score
proposals we retained. A detector that returns almost no confident boxes
lands the probe at LOW, which lets §8 refuse rather than proceed on garbage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cv_agent.profiler.proposals import FrameProposals
from cv_agent.schemas import PixelsOnTarget, ProbeConfidence


_SCORE_THRESHOLD = 0.3
_HIGH_MIN_FRAMES = 20
_HIGH_MIN_PROPOSALS = 30
_MEDIUM_MIN_FRAMES = 8
_MEDIUM_MIN_PROPOSALS = 10


def measure_pixels_on_target(
    frames_with_proposals: Sequence[FrameProposals],
) -> PixelsOnTarget:
    if not frames_with_proposals:
        raise ValueError("measure_pixels_on_target: input must be non-empty")

    heights: list[float] = []
    for fp in frames_with_proposals:
        for p in fp.proposals:
            if p.score >= _SCORE_THRESHOLD:
                heights.append(p.box.height)

    if not heights:
        return PixelsOnTarget(
            p10_px=0.0,
            p50_px=0.0,
            confidence=ProbeConfidence.LOW,
        )

    return PixelsOnTarget(
        p10_px=float(np.percentile(heights, 10)),
        p50_px=float(np.percentile(heights, 50)),
        confidence=_confidence(len(frames_with_proposals), len(heights)),
    )


def _confidence(n_frames: int, n_proposals: int) -> ProbeConfidence:
    if n_frames >= _HIGH_MIN_FRAMES and n_proposals >= _HIGH_MIN_PROPOSALS:
        return ProbeConfidence.HIGH
    if n_frames >= _MEDIUM_MIN_FRAMES and n_proposals >= _MEDIUM_MIN_PROPOSALS:
        return ProbeConfidence.MEDIUM
    return ProbeConfidence.LOW
