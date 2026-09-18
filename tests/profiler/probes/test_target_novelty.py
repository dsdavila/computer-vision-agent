"""Target novelty probe tests."""

from __future__ import annotations

import pytest

from cv_agent.profiler.probes import measure_target_novelty
from cv_agent.schemas import ProbeConfidence, TargetNovelty

from .conftest import frames_with, make_proposal


ONTOLOGY = ("person", "vehicle")


def test_returns_target_novelty_reading():
    fps = frames_with([[make_proposal(label="person", score=0.8)]] * 10)
    r = measure_target_novelty(fps, ONTOLOGY)
    assert isinstance(r, TargetNovelty)


def test_high_scores_yield_high_p50_and_confidence():
    fps = frames_with([[
        make_proposal(label="person", score=0.9),
        make_proposal(label="vehicle", score=0.85),
    ]] * 20)
    r = measure_target_novelty(fps, ONTOLOGY)
    assert r.open_vocab_score_p50 >= 0.85
    assert r.confidence is ProbeConfidence.HIGH


def test_low_scores_downgrade_to_low_confidence():
    # §4.1 / §14: uniformly low scores cannot distinguish "novel targets" from
    # "detector broken" — probe must return LOW so §8 refuses.
    fps = frames_with([[
        make_proposal(label="person", score=0.1),
        make_proposal(label="vehicle", score=0.08),
    ]] * 20)
    r = measure_target_novelty(fps, ONTOLOGY)
    assert r.open_vocab_score_p50 < 0.2
    assert r.confidence is ProbeConfidence.LOW


def test_medium_scores_keep_size_confidence():
    # p50 above the bimodal-failure threshold → confidence keys on sample size.
    fps = frames_with([[
        make_proposal(label="person", score=0.5),
        make_proposal(label="vehicle", score=0.6),
    ]] * 20)
    r = measure_target_novelty(fps, ONTOLOGY)
    assert r.open_vocab_score_p50 >= 0.5
    assert r.confidence is ProbeConfidence.HIGH


def test_no_scores_no_ontology_matches_yields_zero_and_low():
    # Detector proposed nothing that matches the ontology.
    fps = frames_with([[make_proposal(label="dog", score=0.9)]] * 20)
    r = measure_target_novelty(fps, ONTOLOGY)
    assert r.open_vocab_score_p10 == 0.0
    assert r.open_vocab_score_p50 == 0.0
    assert r.confidence is ProbeConfidence.LOW


def test_no_score_threshold_applied():
    # target_novelty must NOT filter by score — the low scores are the signal.
    # 10 proposals all with score 0.15 → p50 = 0.15, not zero.
    fps = frames_with([[make_proposal(label="person", score=0.15)]] * 20)
    r = measure_target_novelty(fps, ONTOLOGY)
    assert r.open_vocab_score_p50 == pytest.approx(0.15)
    # Still LOW because 0.15 < 0.2 (bimodal failure floor).
    assert r.confidence is ProbeConfidence.LOW


def test_proposals_outside_ontology_ignored():
    fps = frames_with([[
        make_proposal(label="person", score=0.9),
        make_proposal(label="airplane", score=0.05),  # not in ontology
    ]] * 20)
    r = measure_target_novelty(fps, ONTOLOGY)
    # Only person scores contribute; p50 should be near 0.9.
    assert r.open_vocab_score_p50 == pytest.approx(0.9)


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        measure_target_novelty([], ONTOLOGY)


def test_empty_ontology_rejected():
    fps = frames_with([[make_proposal()]] * 10)
    with pytest.raises(ValueError):
        measure_target_novelty(fps, [])


def test_confidence_tiers_when_p50_healthy():
    # HIGH: 20 frames × 2 proposals = 40, p50 healthy.
    fps = frames_with([[
        make_proposal(label="person", score=0.7),
        make_proposal(label="vehicle", score=0.8),
    ]] * 20)
    assert measure_target_novelty(fps, ONTOLOGY).confidence is ProbeConfidence.HIGH

    # MEDIUM: 10 frames × 1 proposal.
    fps = frames_with([[make_proposal(label="person", score=0.7)]] * 10)
    assert measure_target_novelty(fps, ONTOLOGY).confidence is ProbeConfidence.MEDIUM
