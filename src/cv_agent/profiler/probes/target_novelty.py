"""Target novelty probe — §4 proposal-dependent property.

Emits p10 and p50 of proposal scores across ontology targets. High scores →
the detector confidently sees targets in the ontology (low novelty). Low
scores → the detector is fumbling (high novelty; may need few-shot or
fine-tune per §14).

**No score threshold.** The other proposal-dependent probes filter out
low-score noise; this one is *about* the score distribution, so filtering
would erase the signal we're measuring.

Proposal-dependent per §4.1, and the sharpest bootstrap-problem case: low
p50 could mean "genuinely novel targets" OR "detector broken", and the
probe cannot tell which. When p50 is very low we degrade to LOW confidence
so §8 refuses rather than proceeding on a signal it cannot interpret. This
is exactly the §14 bimodal-regime tell.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cv_agent.profiler.proposals import FrameProposals
from cv_agent.schemas import ProbeConfidence, TargetNovelty


# p50 below this is treated as "detector may be in the failed half of the
# bimodal regime" (§14) — we cannot distinguish novel-targets from
# broken-detector, so §8 sees LOW and refuses.
_BIMODAL_FAILURE_P50 = 0.2

_HIGH_MIN_FRAMES = 20
_HIGH_MIN_PROPOSALS = 30
_MEDIUM_MIN_FRAMES = 8
_MEDIUM_MIN_PROPOSALS = 10


def measure_target_novelty(
    frames_with_proposals: Sequence[FrameProposals],
    ontology_targets: Sequence[str],
) -> TargetNovelty:
    if not frames_with_proposals:
        raise ValueError("measure_target_novelty: input must be non-empty")
    if not ontology_targets:
        raise ValueError("measure_target_novelty: ontology_targets must be non-empty")

    target_set = set(ontology_targets)
    scores: list[float] = []
    for fp in frames_with_proposals:
        for p in fp.proposals:
            if p.label in target_set:
                scores.append(p.score)

    if not scores:
        # Detector produced no proposals labeled with an ontology target —
        # maximum novelty signal, but also indistinguishable from a broken
        # detector. Report zeros at LOW.
        return TargetNovelty(
            open_vocab_score_p10=0.0,
            open_vocab_score_p50=0.0,
            confidence=ProbeConfidence.LOW,
        )

    p10 = float(np.percentile(scores, 10))
    p50 = float(np.percentile(scores, 50))
    base_confidence = _confidence(len(frames_with_proposals), len(scores))

    # §4.1 / §14 bimodal check: when p50 is very low, we cannot tell "high
    # novelty" from "detector broken", so trust the reading no more than LOW.
    if p50 < _BIMODAL_FAILURE_P50:
        confidence = ProbeConfidence.LOW
    else:
        confidence = base_confidence

    return TargetNovelty(
        open_vocab_score_p10=p10,
        open_vocab_score_p50=p50,
        confidence=confidence,
    )


def _confidence(n_frames: int, n_proposals: int) -> ProbeConfidence:
    if n_frames >= _HIGH_MIN_FRAMES and n_proposals >= _HIGH_MIN_PROPOSALS:
        return ProbeConfidence.HIGH
    if n_frames >= _MEDIUM_MIN_FRAMES and n_proposals >= _MEDIUM_MIN_PROPOSALS:
        return ProbeConfidence.MEDIUM
    return ProbeConfidence.LOW
