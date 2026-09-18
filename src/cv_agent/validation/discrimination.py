"""Probe-discrimination validation — §4.1, §17 M2 gate.

**The claim.** A probe that returns a plausible number on every scene is
worse than a missing probe (§4.1). Each probe must demonstrate on a
held-out labeled set that its metric actually separates the scene classes
it claims to distinguish. Failure to discriminate blocks that probe from
shipping.

**What the harness does.** For a `DiscriminationClaim` — "on the two
classes I care about, my metric should rank the positive class higher" —
plus a set of labeled readings, compute the ROC-AUC of using the metric to
predict the positive class. AUC >= min_auc → the probe passes for this
claim. Below → it fails, and the probe is not fit for its stated purpose.

Structure lives in code from M2; real held-out labeled data lands later
(the sampler + adjudication UI at M4.5/M5). Until then, tests exercise the
mechanics with synthetic labeled readings, and probes can carry candidate
claims next to their implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscriminationClaim:
    """A probe's testable claim.

    - `probe`: the probe under validation (e.g. "lighting").
    - `metric`: the specific metric within the probe (e.g.
      "intensity_variance_over_time"). Documentation only — the harness
      does not resolve it against the probe's schema.
    - `positive_class` / `negative_class`: class labels the metric is
      claimed to separate. Positive is the class where the metric is
      expected to be **higher**.
    - `min_auc`: separation threshold. Passing requires strictly meeting
      or exceeding this. Below is a shipping blocker for this probe."""

    probe: str
    metric: str
    positive_class: str
    negative_class: str
    min_auc: float

    def __post_init__(self) -> None:
        if not (0.5 <= self.min_auc <= 1.0):
            raise ValueError(
                f"DiscriminationClaim.min_auc must be in [0.5, 1.0], got {self.min_auc}"
            )
        if self.positive_class == self.negative_class:
            raise ValueError(
                f"DiscriminationClaim: positive_class ({self.positive_class!r}) "
                "cannot equal negative_class"
            )


@dataclass(frozen=True)
class LabeledReading:
    """One class-labeled numeric reading. Callers extract the specific probe
    metric value; the harness treats readings as opaque floats."""

    class_label: str
    value: float


@dataclass(frozen=True)
class DiscriminationResult:
    """Outcome of evaluating one claim."""

    claim: DiscriminationClaim
    n_positive: int
    n_negative: int
    auc: float
    passed: bool

    @property
    def refused(self) -> bool:
        return not self.passed


def evaluate_claim(
    claim: DiscriminationClaim,
    readings: Sequence[LabeledReading],
) -> DiscriminationResult:
    """Evaluate one claim against labeled readings.

    Readings whose `class_label` matches neither the claim's positive nor
    negative class are ignored — real datasets often carry other labels
    the claim does not speak to. Raises when either side has no readings
    (a claim cannot be evaluated without at least one positive and one
    negative sample)."""

    pos_values = np.array(
        [r.value for r in readings if r.class_label == claim.positive_class],
        dtype=np.float64,
    )
    neg_values = np.array(
        [r.value for r in readings if r.class_label == claim.negative_class],
        dtype=np.float64,
    )

    if pos_values.size == 0 or neg_values.size == 0:
        raise ValueError(
            f"evaluate_claim({claim.probe}.{claim.metric}): need at least one reading "
            f"per class; got {pos_values.size} positive ({claim.positive_class!r}) "
            f"and {neg_values.size} negative ({claim.negative_class!r})"
        )

    auc = _pairwise_auc(pos_values, neg_values)
    return DiscriminationResult(
        claim=claim,
        n_positive=int(pos_values.size),
        n_negative=int(neg_values.size),
        auc=auc,
        passed=auc >= claim.min_auc,
    )


def evaluate_claims(
    claims: Sequence[DiscriminationClaim],
    readings_by_metric: dict[str, Sequence[LabeledReading]],
) -> list[DiscriminationResult]:
    """Evaluate multiple claims at once.

    `readings_by_metric` maps a `metric` name to the labeled readings for
    that metric. Claims for which no readings are provided raise, so
    callers cannot silently forget to feed data for a declared claim."""

    results: list[DiscriminationResult] = []
    for claim in claims:
        if claim.metric not in readings_by_metric:
            raise KeyError(
                f"evaluate_claims: no readings provided for metric {claim.metric!r}"
            )
        results.append(evaluate_claim(claim, readings_by_metric[claim.metric]))
    return results


def _pairwise_auc(pos_values: np.ndarray, neg_values: np.ndarray) -> float:
    """ROC-AUC as the pairwise-rank sum. Equivalent to the probability that
    a randomly chosen positive scores higher than a randomly chosen
    negative; ties count as 0.5."""
    p = pos_values[:, None]
    n = neg_values[None, :]
    greater = np.sum(p > n)
    equal = np.sum(p == n)
    total = pos_values.size * neg_values.size
    return float((greater + 0.5 * equal) / total)
