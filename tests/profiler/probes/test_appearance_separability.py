"""Appearance separability probe tests."""

from __future__ import annotations

import numpy as np
import pytest

from cv_agent.profiler.probes import measure_appearance_separability
from cv_agent.schemas import AppearanceSeparability, ProbeConfidence

from .conftest import frames_with, make_proposal


D = 32  # embedding dim used in these tests


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(D).astype(np.float32)


def test_returns_appearance_separability_reading():
    fps = frames_with([[
        make_proposal(embedding=_emb(0)),
        make_proposal(embedding=_emb(1)),
    ]] * 10)
    r = measure_appearance_separability(fps)
    assert isinstance(r, AppearanceSeparability)


def test_identical_embeddings_give_zero_distance():
    # Every proposal shares the same embedding vector → cosine distance ≈ 0
    # for every pair.
    emb = _emb(42)
    fps = frames_with([[make_proposal(embedding=emb.copy()) for _ in range(3)]] * 10)
    r = measure_appearance_separability(fps)
    assert r.inter_instance_distance_p50 == pytest.approx(0.0, abs=1e-4)


def test_diverse_random_embeddings_have_positive_distance():
    # Fresh random embedding per proposal — in D=32 space these are near-
    # orthogonal, so cosine distance clusters near 1.
    fps = frames_with([[
        make_proposal(embedding=_emb(i))
        for i in range(6)
    ]] * 5)
    r = measure_appearance_separability(fps)
    assert r.inter_instance_distance_p50 > 0.5


def test_no_embeddings_yields_low_confidence_zero():
    fps = frames_with([[make_proposal(embedding=None)] * 3] * 20)
    r = measure_appearance_separability(fps)
    assert r.confidence is ProbeConfidence.LOW
    assert r.inter_instance_distance_p50 == 0.0


def test_low_score_proposals_filtered_out():
    fps = frames_with([[
        make_proposal(embedding=_emb(i), score=0.1)
        for i in range(3)
    ]] * 20)
    r = measure_appearance_separability(fps)
    assert r.confidence is ProbeConfidence.LOW


def test_confidence_tiers():
    # HIGH: 20 frames × 2 proposals with embeddings = 40 embeddings.
    fps = frames_with([[
        make_proposal(embedding=_emb(i)),
        make_proposal(embedding=_emb(i + 100)),
    ] for i in range(20)])
    assert measure_appearance_separability(fps).confidence is ProbeConfidence.HIGH

    # MEDIUM: 10 frames × 1 proposal.
    fps = frames_with([[make_proposal(embedding=_emb(i))] for i in range(10)])
    assert measure_appearance_separability(fps).confidence is ProbeConfidence.MEDIUM

    # LOW: too few.
    fps = frames_with([[make_proposal(embedding=_emb(i))] for i in range(3)])
    assert measure_appearance_separability(fps).confidence is ProbeConfidence.LOW


def test_single_embedding_yields_low_confidence():
    # Only one embedding → no pair to compare.
    fps = frames_with([[make_proposal(embedding=_emb(0))]] + [[]] * 19)
    r = measure_appearance_separability(fps)
    assert r.confidence is ProbeConfidence.LOW
    assert r.inter_instance_distance_p50 == 0.0


def test_zero_embedding_ignored():
    # Zero vectors have undefined cosine — must be dropped, not crash. Set up
    # a scene where most embeddings are zero so we can assert the drop
    # happened (few valid embeddings remain → LOW confidence).
    zero = np.zeros(D, dtype=np.float32)
    fps = frames_with(
        [[make_proposal(embedding=zero)] for _ in range(19)]
        + [[make_proposal(embedding=_emb(1))]]
    )
    r = measure_appearance_separability(fps)
    # Only one non-zero embedding survives; can't compute a pair → LOW.
    assert r.confidence is ProbeConfidence.LOW
    assert r.inter_instance_distance_p50 == 0.0


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        measure_appearance_separability([])
