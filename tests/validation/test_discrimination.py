"""Probe-discrimination harness tests.

Two layers of tests:
- Mechanics: AUC computation, claim invariants, evaluate_claim / evaluate_claims.
- End-to-end: build a labeled synthetic dataset, run a probe on each item,
  extract the target metric, feed the labeled readings to the harness.
  Confirms the whole loop works with a real probe.
"""

from __future__ import annotations

import numpy as np
import pytest

from cv_agent.profiler.probes import measure_lighting
from cv_agent.profiler.video import Frame
from cv_agent.validation import (
    DiscriminationClaim,
    LabeledReading,
    evaluate_claim,
    evaluate_claims,
)


# --- claim invariants -------------------------------------------------------


def test_claim_rejects_min_auc_below_chance():
    with pytest.raises(ValueError):
        DiscriminationClaim(
            probe="lighting",
            metric="intensity_variance_over_time",
            positive_class="varying",
            negative_class="stable",
            min_auc=0.4,
        )


def test_claim_rejects_min_auc_above_one():
    with pytest.raises(ValueError):
        DiscriminationClaim(
            probe="lighting",
            metric="intensity_variance_over_time",
            positive_class="varying",
            negative_class="stable",
            min_auc=1.5,
        )


def test_claim_rejects_positive_equals_negative():
    with pytest.raises(ValueError):
        DiscriminationClaim(
            probe="lighting",
            metric="x",
            positive_class="a",
            negative_class="a",
            min_auc=0.7,
        )


# --- AUC computation --------------------------------------------------------


def _claim(min_auc: float = 0.7) -> DiscriminationClaim:
    return DiscriminationClaim(
        probe="test",
        metric="m",
        positive_class="pos",
        negative_class="neg",
        min_auc=min_auc,
    )


def test_perfect_separation_yields_auc_one():
    readings = [LabeledReading("pos", v) for v in [10.0, 11.0, 12.0]] + [
        LabeledReading("neg", v) for v in [0.0, 1.0, 2.0]
    ]
    r = evaluate_claim(_claim(), readings)
    assert r.auc == pytest.approx(1.0)
    assert r.passed is True


def test_reverse_separation_yields_auc_zero():
    # Positive values are all LOWER than negatives → AUC = 0.
    readings = [LabeledReading("pos", v) for v in [0.0, 1.0, 2.0]] + [
        LabeledReading("neg", v) for v in [10.0, 11.0, 12.0]
    ]
    r = evaluate_claim(_claim(), readings)
    assert r.auc == pytest.approx(0.0)
    assert r.passed is False


def test_no_separation_yields_auc_half():
    readings = [LabeledReading("pos", v) for v in [1.0, 2.0, 3.0]] + [
        LabeledReading("neg", v) for v in [1.0, 2.0, 3.0]
    ]
    r = evaluate_claim(_claim(), readings)
    assert r.auc == pytest.approx(0.5)
    assert r.passed is False


def test_ties_count_as_half():
    # All positive == all negative → AUC = 0.5.
    readings = [LabeledReading("pos", 5.0)] * 3 + [LabeledReading("neg", 5.0)] * 3
    r = evaluate_claim(_claim(), readings)
    assert r.auc == pytest.approx(0.5)


def test_result_reports_sample_counts():
    readings = [
        LabeledReading("pos", 1.0),
        LabeledReading("pos", 2.0),
        LabeledReading("neg", 0.0),
        LabeledReading("neg", 0.5),
        LabeledReading("neg", 0.9),
    ]
    r = evaluate_claim(_claim(), readings)
    assert r.n_positive == 2
    assert r.n_negative == 3


def test_readings_with_unrelated_labels_are_ignored():
    # Claim covers pos/neg only; a "other" label should be dropped, not
    # crash the evaluator.
    readings = [
        LabeledReading("pos", 5.0),
        LabeledReading("neg", 0.0),
        LabeledReading("other", 99.0),
    ]
    r = evaluate_claim(_claim(), readings)
    assert r.n_positive == 1 and r.n_negative == 1


def test_missing_class_raises():
    with pytest.raises(ValueError, match="need at least one reading"):
        evaluate_claim(_claim(), [LabeledReading("pos", 1.0)])
    with pytest.raises(ValueError, match="need at least one reading"):
        evaluate_claim(_claim(), [LabeledReading("neg", 1.0)])


# --- evaluate_claims (multi-claim) ------------------------------------------


def test_evaluate_claims_runs_each():
    claims = (
        _claim(min_auc=0.8),
        DiscriminationClaim(
            probe="test",
            metric="m2",
            positive_class="pos",
            negative_class="neg",
            min_auc=0.9,
        ),
    )
    readings_m1 = [LabeledReading("pos", 2.0), LabeledReading("neg", 1.0)]
    readings_m2 = [LabeledReading("pos", 3.0), LabeledReading("neg", 1.0)]
    results = evaluate_claims(claims, {"m": readings_m1, "m2": readings_m2})
    assert len(results) == 2
    assert results[0].auc == 1.0
    assert results[1].auc == 1.0


def test_evaluate_claims_raises_on_missing_metric():
    claim = _claim()
    with pytest.raises(KeyError):
        evaluate_claims([claim], {})


# --- end-to-end with a real probe -------------------------------------------


def _grey_frame(index: int, gray: int, size: tuple[int, int] = (48, 64)) -> Frame:
    arr = np.full((size[0], size[1], 3), gray, dtype=np.uint8)
    return Frame(array=arr, index=index, timestamp_s=index / 30.0)


def _varying_lighting_frames() -> list[Frame]:
    """A 20-frame set with linearly ramping intensity — high variance."""
    return [_grey_frame(i, int(round(i * 255 / 19))) for i in range(20)]


def _stable_lighting_frames() -> list[Frame]:
    """20 frames at constant intensity — near-zero variance."""
    return [_grey_frame(i, 128) for i in range(20)]


def test_lighting_probe_discriminates_varying_from_stable():
    """End-to-end validation of the lighting probe's central claim: its
    intensity_variance_over_time reading separates varying from stable
    lighting."""

    claim = DiscriminationClaim(
        probe="lighting",
        metric="intensity_variance_over_time",
        positive_class="varying",
        negative_class="stable",
        min_auc=0.9,
    )

    # 5 varying + 5 stable synthetic clips. The probe runs on each; the
    # extracted metric is the labeled reading fed to the harness.
    readings: list[LabeledReading] = []
    for _ in range(5):
        r = measure_lighting(_varying_lighting_frames())
        readings.append(LabeledReading("varying", r.intensity_variance_over_time))
    for _ in range(5):
        r = measure_lighting(_stable_lighting_frames())
        readings.append(LabeledReading("stable", r.intensity_variance_over_time))

    result = evaluate_claim(claim, readings)
    assert result.passed
    assert result.auc == pytest.approx(1.0)
    assert result.n_positive == 5 and result.n_negative == 5


def test_lighting_probe_saturation_claim():
    """Second lighting claim: saturation_clipping_fraction separates clipped
    (all-white) from mid-grey scenes."""

    claim = DiscriminationClaim(
        probe="lighting",
        metric="saturation_clipping_fraction",
        positive_class="clipped",
        negative_class="mid_grey",
        min_auc=0.95,
    )
    readings: list[LabeledReading] = []
    for _ in range(5):
        r = measure_lighting([_grey_frame(i, 255) for i in range(10)])
        readings.append(LabeledReading("clipped", r.saturation_clipping_fraction))
    for _ in range(5):
        r = measure_lighting([_grey_frame(i, 128) for i in range(10)])
        readings.append(LabeledReading("mid_grey", r.saturation_clipping_fraction))

    result = evaluate_claim(claim, readings)
    assert result.passed
    assert result.auc == pytest.approx(1.0)
