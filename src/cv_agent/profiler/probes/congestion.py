"""Congestion probe — §4 proposal-dependent property.

Emits three signals about how crowded the scene is:

- detections_per_frame_mean: average count of retained proposals per frame.
- mean_pairwise_iou: within each frame, mean IoU over all box pairs; then
  mean across frames.
- occlusion_rate: fraction of retained proposals whose max IoU with any
  other proposal in the same frame is ≥ 0.3. A proxy for occlusion without
  depth or tracking.

Proposal-dependent per §4.1: confidence keys on retained-proposal density.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cv_agent.profiler.proposals import Box, FrameProposals
from cv_agent.schemas import Congestion, ProbeConfidence


_SCORE_THRESHOLD = 0.3
_OCCLUSION_IOU = 0.3
_HIGH_MIN_FRAMES = 20
_HIGH_MIN_PROPOSALS = 30
_MEDIUM_MIN_FRAMES = 8
_MEDIUM_MIN_PROPOSALS = 10


def measure_congestion(
    frames_with_proposals: Sequence[FrameProposals],
) -> Congestion:
    if not frames_with_proposals:
        raise ValueError("measure_congestion: input must be non-empty")

    per_frame_counts: list[int] = []
    per_frame_mean_iou: list[float] = []
    occluded_total = 0
    proposals_total = 0

    for fp in frames_with_proposals:
        boxes: list[Box] = [
            p.box for p in fp.proposals if p.score >= _SCORE_THRESHOLD
        ]
        per_frame_counts.append(len(boxes))
        proposals_total += len(boxes)

        if len(boxes) < 2:
            per_frame_mean_iou.append(0.0)
            continue

        pair_ious: list[float] = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                pair_ious.append(boxes[i].iou(boxes[j]))
        per_frame_mean_iou.append(float(np.mean(pair_ious)))

        for i, b1 in enumerate(boxes):
            max_iou_for_i = 0.0
            for j, b2 in enumerate(boxes):
                if i == j:
                    continue
                iou = b1.iou(b2)
                if iou > max_iou_for_i:
                    max_iou_for_i = iou
            if max_iou_for_i >= _OCCLUSION_IOU:
                occluded_total += 1

    detections_mean = float(np.mean(per_frame_counts))
    mean_pairwise = float(np.mean(per_frame_mean_iou))
    occlusion_rate = (
        occluded_total / proposals_total if proposals_total > 0 else 0.0
    )

    return Congestion(
        detections_per_frame_mean=detections_mean,
        mean_pairwise_iou=mean_pairwise,
        occlusion_rate=occlusion_rate,
        confidence=_confidence(len(frames_with_proposals), proposals_total),
    )


def _confidence(n_frames: int, n_proposals: int) -> ProbeConfidence:
    if n_frames >= _HIGH_MIN_FRAMES and n_proposals >= _HIGH_MIN_PROPOSALS:
        return ProbeConfidence.HIGH
    if n_frames >= _MEDIUM_MIN_FRAMES and n_proposals >= _MEDIUM_MIN_PROPOSALS:
        return ProbeConfidence.MEDIUM
    return ProbeConfidence.LOW
