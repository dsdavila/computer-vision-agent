"""Tier-3 scorer — GT eval set + scoring function, per-stratum CIs (§6, §7.2).

§17 M3 calls tier 3 "what makes a multi-topology planner legitimate": tier 3
is the only tier whose scores are comparable across topologies. Tier 0
filters; tier 1 ranks within a topology (§6); tier 3 ranks across.

§7.2 discipline: the API returns per-stratum CIs, **never a scalar pooled
over strata**. A pooled scalar reintroduces the false-confidence trap
stratification exists to eliminate. Where a headline number is genuinely
wanted, `pooled_with_spread()` returns the pooled value AND the per-stratum
spread alongside — never one without the other.

Scorer contract: `(pipeline_output, gt, eval_set) -> tuple[StratumScore]`.
Named scorers register into a global registry keyed by `EvalSet.scorer_id`;
`compute_tier3` looks the scorer up so the EvalSet is fully self-describing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from cv_agent.profiler.proposals import Box, FrameProposals
from cv_agent.schemas import EvalSet
from cv_agent.scoring.coco import CocoGroundTruth, GTBox
from cv_agent.scoring.pipeline import PipelineOutput
from cv_agent.scoring.topology import Topology


# --- score types -----------------------------------------------------------


@dataclass(frozen=True)
class StratumScore:
    """One stratum's score + 95% Wilson CI + item count."""

    stratum: str
    value: float
    ci_low: float
    ci_high: float
    n_items: int


@dataclass(frozen=True)
class Tier3Score:
    """§7.2 per-stratum readout. Deliberately no pooled scalar — callers
    who want one call pooled_with_spread(), which returns the per-stratum
    spread alongside."""

    topology: Topology
    metric: str
    scorer_id: str
    eval_set_name: str
    strata: tuple[StratumScore, ...]

    def pooled_with_spread(self) -> tuple[float, float, float]:
        """Return (pooled_value, min_stratum_value, max_stratum_value).

        The pooled value is a sample-weighted mean across strata; the spread
        is min/max across strata. §7.2: never return one without the other,
        and never let the pooled scalar be the only visible number."""
        if not self.strata:
            return (0.0, 0.0, 0.0)
        total_n = sum(s.n_items for s in self.strata)
        if total_n == 0:
            return (0.0, 0.0, 0.0)
        pooled = sum(s.value * s.n_items for s in self.strata) / total_n
        values = [s.value for s in self.strata]
        return (pooled, min(values), max(values))

    def stratum(self, name: str) -> StratumScore:
        for s in self.strata:
            if s.stratum == name:
                return s
        raise KeyError(f"Tier3Score has no stratum {name!r}")


# --- scoring function contract + registry ---------------------------------


ScoringFunction = Callable[
    [PipelineOutput, CocoGroundTruth, EvalSet],
    tuple[StratumScore, ...],
]


_SCORER_REGISTRY: dict[str, tuple[str, ScoringFunction]] = {}


def register_scorer(
    scorer_id: str, *, metric_name: str, fn: ScoringFunction
) -> None:
    """Register a named scorer. `metric_name` is what Tier3Score.metric is
    set to (e.g. "recall@0.5") — used by the ledger narrative and by tier-3
    comparisons."""
    _SCORER_REGISTRY[scorer_id] = (metric_name, fn)


def get_scorer(scorer_id: str) -> tuple[str, ScoringFunction]:
    if scorer_id not in _SCORER_REGISTRY:
        known = ", ".join(sorted(_SCORER_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown scorer {scorer_id!r}. Registered: {known}. "
            "Register via cv_agent.scoring.tier3.register_scorer()."
        )
    return _SCORER_REGISTRY[scorer_id]


# --- public entrypoint ----------------------------------------------------


def compute_tier3(
    topology: Topology,
    pipeline_output: PipelineOutput,
    eval_set: EvalSet,
    gt: CocoGroundTruth,
) -> Tier3Score:
    """Score a pipeline's output against an EvalSet. Looks up the scorer
    named in `eval_set.scorer_id`; the EvalSet is fully self-describing."""
    metric_name, scorer = get_scorer(eval_set.scorer_id)
    strata_scores = scorer(pipeline_output, gt, eval_set)
    return Tier3Score(
        topology=topology,
        metric=metric_name,
        scorer_id=eval_set.scorer_id,
        eval_set_name=eval_set.name,
        strata=strata_scores,
    )


# --- comparison / dominance -----------------------------------------------


def compare_tier3(
    a: Tier3Score, b: Tier3Score, *, on_stratum: str
) -> int:
    """Compare two Tier3Scores on one stratum. Returns 1 if `a` is better,
    -1 if worse, 0 if equal. Higher value is better (recall / precision /
    F1 etc.); scorers that measure "lower is better" (false-alarm rate,
    fragmentation) should invert their reported values so this convention
    holds.

    Cross-topology comparison is legitimate here (§17 M3) — this is what
    tier 3 exists for. Different topologies scored on the same EvalSet are
    directly comparable."""
    if a.scorer_id != b.scorer_id:
        raise ValueError(
            f"compare_tier3: scorer mismatch ({a.scorer_id!r} vs {b.scorer_id!r})"
        )
    if a.eval_set_name != b.eval_set_name:
        raise ValueError(
            f"compare_tier3: eval-set mismatch "
            f"({a.eval_set_name!r} vs {b.eval_set_name!r})"
        )

    va = a.stratum(on_stratum).value
    vb = b.stratum(on_stratum).value
    if va == vb:
        return 0
    return 1 if va > vb else -1


def dominates(a: Tier3Score, b: Tier3Score) -> bool:
    """Pareto dominance. Returns True iff `a` is >= `b` on every shared
    stratum AND strictly greater on at least one. Higher-is-better
    convention; see compare_tier3."""
    if a.scorer_id != b.scorer_id or a.eval_set_name != b.eval_set_name:
        return False

    a_strata = {s.stratum: s.value for s in a.strata}
    b_strata = {s.stratum: s.value for s in b.strata}
    shared = set(a_strata) & set(b_strata)
    if not shared:
        return False

    strictly_greater = False
    for name in shared:
        if a_strata[name] < b_strata[name]:
            return False
        if a_strata[name] > b_strata[name]:
            strictly_greater = True
    return strictly_greater


# --- Wilson score CI ------------------------------------------------------


_Z_95 = 1.959963984540054  # 95% CI z-score


def _wilson_ci(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Robust at
    small n; standard choice for recall/precision-style metrics."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = _Z_95 * _Z_95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = _Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --- default scorer: recall@IoU 0.5 ---------------------------------------


def _iou(a: Box, b: Box) -> float:
    return a.iou(b)


def _match_and_count(
    gt_boxes: tuple[GTBox, ...],
    predicted: tuple[FrameProposals, ...] | tuple[Box, ...],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedy per-frame matching. Returns (tp, fn, fp).

    Matching is class-aware: a prediction only matches a GT with the same
    label. Each GT can be matched by at most one prediction; each prediction
    can be matched to at most one GT. Higher-IoU predictions match first."""
    if isinstance(predicted, tuple) and predicted and isinstance(predicted[0], Box):
        # Shouldn't happen in the current design, but keep the signature simple.
        pred_boxes_and_labels: list[tuple[Box, str]] = [(b, "") for b in predicted]  # type: ignore[misc]
    else:
        pred_boxes_and_labels = []
        for pred_fp in predicted:  # type: ignore[assignment]
            # `predicted` here is meant to be prediction Proposals for ONE
            # image; callers pre-slice. In this scorer, we call this per
            # image and pass a list of (Box, label) directly.
            raise NotImplementedError

    return _match_boxes_and_count(gt_boxes, pred_boxes_and_labels, iou_threshold)


def _match_boxes_and_count(
    gt_boxes: tuple[GTBox, ...],
    predicted: list[tuple[Box, str]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedy class-aware matching. `predicted` is a list of (box, label)
    for one image."""
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(predicted)

    # Score all (gt, pred) pairs, sort descending by IoU, greedily accept.
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, (pbox, plabel) in enumerate(predicted):
            if plabel != gt.label:
                continue
            iou = _iou(gt.box, pbox)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda t: -t[0])
    for _, gi, pi in candidates:
        if gt_matched[gi] or pred_matched[pi]:
            continue
        gt_matched[gi] = True
        pred_matched[pi] = True

    tp = sum(gt_matched)
    fn = len(gt_boxes) - tp
    fp = sum(1 for m in pred_matched if not m)
    return tp, fn, fp


def recall_at_iou_0_5(
    pipeline_output: PipelineOutput,
    gt: CocoGroundTruth,
    eval_set: EvalSet,
) -> tuple[StratumScore, ...]:
    """Per-stratum recall at IoU >= 0.5, class-aware. Ships as the v0
    default scorer; registered under scorer_id="recall@0.5"."""

    # Index predictions by image_id (assumed to equal Frame.index).
    preds_by_image: dict[int, list[tuple[Box, str]]] = {}
    for fp in pipeline_output.frame_proposals:
        preds_by_image[fp.frame.index] = [
            (p.box, p.label) for p in fp.proposals
        ]

    strata_names = tuple(s.name for s in eval_set.strata)
    per_stratum_tp: dict[str, int] = {s: 0 for s in strata_names}
    per_stratum_n: dict[str, int] = {s: 0 for s in strata_names}

    for image_id_str, stratum_name in eval_set.item_strata.items():
        try:
            image_id = int(image_id_str)
        except ValueError:
            # Non-integer image ids (e.g. MOT clip names) don't apply here;
            # the recall scorer is COCO-only, so skip.
            continue
        gt_boxes = gt.boxes_by_image.get(image_id, ())
        preds = preds_by_image.get(image_id, [])
        tp, fn, _fp = _match_boxes_and_count(gt_boxes, preds, iou_threshold=0.5)
        n = tp + fn
        per_stratum_tp[stratum_name] = per_stratum_tp.get(stratum_name, 0) + tp
        per_stratum_n[stratum_name] = per_stratum_n.get(stratum_name, 0) + n

    results: list[StratumScore] = []
    for name in strata_names:
        n = per_stratum_n[name]
        tp = per_stratum_tp[name]
        if n == 0:
            results.append(
                StratumScore(
                    stratum=name, value=0.0, ci_low=0.0, ci_high=0.0, n_items=0
                )
            )
            continue
        recall = tp / n
        ci_low, ci_high = _wilson_ci(tp, n)
        results.append(
            StratumScore(
                stratum=name,
                value=recall,
                ci_low=ci_low,
                ci_high=ci_high,
                n_items=n,
            )
        )
    return tuple(results)


register_scorer("recall@0.5", metric_name="recall@0.5", fn=recall_at_iou_0_5)
