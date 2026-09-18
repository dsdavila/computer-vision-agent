"""Tier-1 label-free proxy scorer — §6 within-topology comparison only.

§6 central risk: label-free proxies rank configurations WITHIN an
equivalence class (same detector family, same class vocabulary). They do
NOT rank across classes. Treating them as globally comparable will
confidently select the wrong architecture.

This module enforces that in code, not just in documentation:

- Tier1Score carries the Topology it was computed against.
- Tier1Score explicitly raises on `<`, `>`, `<=`, `>=` — the "not
  orderable" language in §6 becomes a Python-level error, not a
  convention. The exception message names §6 and points to
  compare_tier1() which does the right thing.
- compare_tier1(a, b, on=metric) checks a.topology == b.topology and
  raises CrossTopologyComparisonError if they differ.

Four proxies (§6): short_track_ratio, conf_p50_margin,
near_threshold_fraction, track_count_stability. Direction of "better" is
baked into compare_tier1 per metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cv_agent.scoring.pipeline import PipelineOutput
from cv_agent.scoring.topology import Topology


# --- constants ------------------------------------------------------------


_SHORT_TRACK_MAX_FRAMES = 5           # Tracks under 5 frames count as "short".
_NEAR_THRESHOLD_MARGIN = 0.1          # |score - threshold| < margin → near.

_VALID_METRICS: tuple[str, ...] = (
    "short_track_ratio",
    "conf_p50_margin",
    "near_threshold_fraction",
    "track_count_stability",
)

# Direction: metrics where LOWER is better vs HIGHER is better.
_LOWER_IS_BETTER = frozenset({"short_track_ratio", "near_threshold_fraction"})


# --- errors ---------------------------------------------------------------


class CrossTopologyComparisonError(ValueError):
    """§6: tier-1 scores are not comparable across topologies."""


# --- score type -----------------------------------------------------------


@dataclass(frozen=True)
class Tier1Score:
    """§6 label-free proxy readout, keyed to one Topology.

    Deliberately not orderable. To compare, use compare_tier1(a, b, on=<metric>)
    which enforces the topology-match invariant."""

    topology: Topology
    short_track_ratio: float
    conf_p50_margin: float
    near_threshold_fraction: float
    track_count_stability: float

    def _cross_topology_ordering_error(self) -> TypeError:
        return TypeError(
            "Tier1Score is not orderable — §6 forbids cross-topology "
            "comparison. Use compare_tier1(a, b, on=<metric>) to compare "
            "two scores within the same topology."
        )

    def __lt__(self, other: Any) -> bool:
        raise self._cross_topology_ordering_error()

    def __le__(self, other: Any) -> bool:
        raise self._cross_topology_ordering_error()

    def __gt__(self, other: Any) -> bool:
        raise self._cross_topology_ordering_error()

    def __ge__(self, other: Any) -> bool:
        raise self._cross_topology_ordering_error()


# --- public entrypoints ---------------------------------------------------


def compute_tier1(topology: Topology, output: PipelineOutput) -> Tier1Score:
    """Compute all four §6 proxies for one topology+config's pipeline output."""
    return Tier1Score(
        topology=topology,
        short_track_ratio=_short_track_ratio(output),
        conf_p50_margin=_conf_p50_margin(output),
        near_threshold_fraction=_near_threshold_fraction(output),
        track_count_stability=_track_count_stability(output),
    )


def compare_tier1(a: Tier1Score, b: Tier1Score, *, on: str) -> int:
    """Compare two Tier1Scores on one metric. Returns 1 if `a` is better
    than `b`, -1 if worse, 0 if equal. Direction of "better" is metric-
    specific (see _LOWER_IS_BETTER).

    Raises CrossTopologyComparisonError when a.topology != b.topology.
    §6: tier-1 scores are not comparable across topologies. Callers who
    need cross-topology selection go through tier 3."""
    if a.topology != b.topology:
        raise CrossTopologyComparisonError(
            f"Tier-1 scores not comparable across topologies (§6). "
            f"a={a.topology.describe()!r} vs b={b.topology.describe()!r}."
        )
    if on not in _VALID_METRICS:
        raise ValueError(
            f"compare_tier1: unknown metric {on!r}; expected one of {_VALID_METRICS}"
        )

    va = getattr(a, on)
    vb = getattr(b, on)
    if va == vb:
        return 0
    a_better = va < vb if on in _LOWER_IS_BETTER else va > vb
    return 1 if a_better else -1


# --- proxy implementations -----------------------------------------------


def _short_track_ratio(output: PipelineOutput) -> float:
    """Fraction of tracks with lifespan below _SHORT_TRACK_MAX_FRAMES.
    Empty tracks -> 0.0 (no fragmentation observable)."""
    if not output.tracks:
        return 0.0
    short = sum(1 for t in output.tracks if t.lifespan_frames < _SHORT_TRACK_MAX_FRAMES)
    return short / len(output.tracks)


def _conf_p50_margin(output: PipelineOutput) -> float:
    """Median detection confidence minus the detector threshold. Positive
    means confident detections above threshold; near zero means the median
    detection sits right at the decision boundary. Empty output -> 0.0."""
    scores = [
        p.score
        for fp in output.frame_proposals
        for p in fp.proposals
    ]
    if not scores:
        return 0.0
    return float(np.median(scores) - output.detector_threshold)


def _near_threshold_fraction(output: PipelineOutput) -> float:
    """Fraction of detections whose score is within _NEAR_THRESHOLD_MARGIN
    of the detector threshold. High values mean the threshold is uncertain
    (many detections could go either way). Empty output -> 0.0."""
    scores = [
        p.score
        for fp in output.frame_proposals
        for p in fp.proposals
    ]
    if not scores:
        return 0.0
    near = sum(
        1 for s in scores if abs(s - output.detector_threshold) < _NEAR_THRESHOLD_MARGIN
    )
    return near / len(scores)


def _track_count_stability(output: PipelineOutput) -> float:
    """1 - coefficient-of-variation of per-frame track count, clamped to
    [0, 1]. High values → stable track count over time (few births/deaths
    per unit time). Empty output -> 1.0 (nothing to be unstable)."""
    per_frame_count = _tracks_per_frame(output)
    if not per_frame_count:
        return 1.0
    counts = np.array(per_frame_count, dtype=np.float64)
    mean = counts.mean()
    if mean == 0:
        return 1.0
    cv = counts.std() / mean
    return float(max(0.0, 1.0 - cv))


def _tracks_per_frame(output: PipelineOutput) -> list[int]:
    """Per-frame track count over the frames the pipeline saw. Uses the
    frame_proposals to enumerate frames (they're the pipeline's view of
    the video); tracks are counted per frame_index."""
    if not output.frame_proposals:
        return []
    frame_indices = [fp.frame.index for fp in output.frame_proposals]
    counts: dict[int, int] = {i: 0 for i in frame_indices}
    for track in output.tracks:
        for tf in track.frames:
            if tf.frame_index in counts:
                counts[tf.frame_index] += 1
    return [counts[i] for i in frame_indices]
