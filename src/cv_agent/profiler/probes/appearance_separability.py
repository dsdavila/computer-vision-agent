"""Appearance separability probe — §4 proposal-dependent property.

Emits p50 of pairwise cosine distances between proposal embeddings. High →
targets look visually distinct (ReID can separate identities); low → targets
look alike (ReID gains little).

Chicken-and-egg avoidance: §4 names this "inter-instance embedding distance
on sampled crops", but the profiler runs before any tracker, so we cannot
group crops by instance. We proxy with all pairs of proposal embeddings —
if the scene contains one type of target, embeddings cluster tight; if
mixed, they spread. Fit for scene classification.

Proposal-dependent per §4.1: confidence downgrades sharply when embeddings
are missing (a detector that produces no appearance features cannot inform
ReID choice, no matter how many boxes it draws).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cv_agent.profiler.proposals import FrameProposals
from cv_agent.schemas import AppearanceSeparability, ProbeConfidence


_SCORE_THRESHOLD = 0.3
_MAX_PAIRS = 500  # cap on pairs to keep the probe O(N * D) rather than O(N^2 * D)
_SAMPLING_SEED = 0
_HIGH_MIN_FRAMES = 20
_HIGH_MIN_EMBEDDINGS = 30
_MEDIUM_MIN_FRAMES = 8
_MEDIUM_MIN_EMBEDDINGS = 10


def measure_appearance_separability(
    frames_with_proposals: Sequence[FrameProposals],
) -> AppearanceSeparability:
    if not frames_with_proposals:
        raise ValueError("measure_appearance_separability: input must be non-empty")

    normalized: list[np.ndarray] = []
    for fp in frames_with_proposals:
        for p in fp.proposals:
            if p.score < _SCORE_THRESHOLD or p.embedding is None:
                continue
            emb = np.asarray(p.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                normalized.append(emb / norm)

    if len(normalized) < 2:
        return AppearanceSeparability(
            inter_instance_distance_p50=0.0,
            confidence=ProbeConfidence.LOW,
        )

    emb_arr = np.stack(normalized)
    n = emb_arr.shape[0]
    total_pairs = n * (n - 1) // 2

    if total_pairs <= _MAX_PAIRS:
        i_idx, j_idx = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(_SAMPLING_SEED)
        i_idx = rng.integers(0, n, size=_MAX_PAIRS * 2)
        j_idx = rng.integers(0, n, size=_MAX_PAIRS * 2)
        mask = i_idx != j_idx
        i_idx = i_idx[mask][:_MAX_PAIRS]
        j_idx = j_idx[mask][:_MAX_PAIRS]

    sims = np.sum(emb_arr[i_idx] * emb_arr[j_idx], axis=1)
    distances = 1.0 - sims

    return AppearanceSeparability(
        inter_instance_distance_p50=float(np.percentile(distances, 50)),
        confidence=_confidence(len(frames_with_proposals), n),
    )


def _confidence(n_frames: int, n_embeddings: int) -> ProbeConfidence:
    if n_frames >= _HIGH_MIN_FRAMES and n_embeddings >= _HIGH_MIN_EMBEDDINGS:
        return ProbeConfidence.HIGH
    if n_frames >= _MEDIUM_MIN_FRAMES and n_embeddings >= _MEDIUM_MIN_EMBEDDINGS:
        return ProbeConfidence.MEDIUM
    return ProbeConfidence.LOW
