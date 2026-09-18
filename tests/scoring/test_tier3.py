"""Tier-3 scorer tests — COCO loading, recall@0.5, per-stratum CIs,
cross-topology comparison, dominance, pooled-with-spread."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cv_agent.profiler.proposals import Box, FrameProposals, Proposal
from cv_agent.profiler.video import Frame
from cv_agent.schemas import (
    EvalSet,
    GroundTruthFile,
    GroundTruthFormat,
    StratumLabel,
)
from cv_agent.scoring import (
    PipelineOutput,
    Tier3Score,
    compare_tier3,
    compute_tier3,
    dominates,
    load_coco,
    recall_at_iou_0_5,
)

from .conftest import make_detector, make_topology


# --- COCO loader -----------------------------------------------------------


def _write_coco(path: Path, images: list[dict], annotations: list[dict], categories: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}
        ),
        encoding="utf-8",
    )


def test_load_coco_indexes_boxes_by_image(tmp_path: Path):
    p = tmp_path / "gt.json"
    _write_coco(
        p,
        images=[{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 60, 20, 30]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 10, 10]},
        ],
        categories=[{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}],
    )
    gt = load_coco(p)
    assert len(gt.boxes_by_image[1]) == 2
    assert len(gt.boxes_by_image[2]) == 1
    assert gt.categories[1] == "person"
    assert gt.boxes_by_image[1][0].label == "person"
    b = gt.boxes_by_image[1][0].box
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 40, 60)  # x, y, x+w, y+h


def test_load_coco_skips_degenerate_annotations(tmp_path: Path):
    p = tmp_path / "gt.json"
    _write_coco(
        p,
        images=[{"id": 1, "file_name": "a.jpg"}],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 10]},  # w=0
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ],
        categories=[{"id": 1, "name": "person"}],
    )
    gt = load_coco(p)
    assert len(gt.boxes_by_image[1]) == 1  # degenerate skipped


# --- fixtures for tier-3 --------------------------------------------------


def _frame(index: int) -> Frame:
    return Frame(
        array=np.zeros((10, 10, 3), dtype=np.uint8),
        index=index,
        timestamp_s=index / 30.0,
    )


def _pred(frame_index: int, box: Box, label: str = "person", score: float = 0.9) -> FrameProposals:
    return FrameProposals(
        frame=_frame(frame_index),
        proposals=(Proposal(box=box, label=label, score=score),),
    )


def _pipeline(frame_proposals: tuple[FrameProposals, ...] = ()) -> PipelineOutput:
    return PipelineOutput(
        frame_proposals=frame_proposals,
        tracks=(),
        detector_threshold=0.4,
    )


def _eval_set(
    strata: tuple[StratumLabel, ...],
    item_strata: dict[str, str],
    scorer_id: str = "recall@0.5",
    name: str = "test_eval",
) -> EvalSet:
    return EvalSet(
        name=name,
        ground_truth=GroundTruthFile(path="dummy.json", format=GroundTruthFormat.COCO),
        strata=strata,
        item_strata=item_strata,
        scorer_id=scorer_id,
        catalog_version="v0-2026-09-17",
        task_contract_name="test_task",
    )


def _gt_from_dict(gt_by_image: dict[int, list[tuple[Box, str]]]):
    """Build a CocoGroundTruth directly from a mapping (avoids writing JSON)."""
    from cv_agent.scoring.coco import CocoGroundTruth, GTBox

    return CocoGroundTruth(
        boxes_by_image={
            image_id: tuple(GTBox(box=b, label=lab) for b, lab in items)
            for image_id, items in gt_by_image.items()
        },
        categories={},
    )


# --- recall@0.5 core behavior ---------------------------------------------


def test_perfect_predictions_yield_recall_one():
    # Two images, one GT each; predictions match perfectly.
    gt = _gt_from_dict({
        1: [(Box(x1=0, y1=0, x2=20, y2=40), "person")],
        2: [(Box(x1=10, y1=10, x2=30, y2=50), "person")],
    })
    preds = (
        _pred(1, Box(x1=0, y1=0, x2=20, y2=40)),
        _pred(2, Box(x1=10, y1=10, x2=30, y2=50)),
    )
    eval_set = _eval_set(
        strata=(StratumLabel(name="all"),),
        item_strata={"1": "all", "2": "all"},
    )
    score = compute_tier3(make_topology(), _pipeline(preds), eval_set, gt)
    all_stratum = score.stratum("all")
    assert all_stratum.value == pytest.approx(1.0)
    assert all_stratum.n_items == 2


def test_no_predictions_yield_recall_zero():
    gt = _gt_from_dict({1: [(Box(x1=0, y1=0, x2=20, y2=40), "person")]})
    eval_set = _eval_set(
        strata=(StratumLabel(name="all"),),
        item_strata={"1": "all"},
    )
    score = compute_tier3(make_topology(), _pipeline(()), eval_set, gt)
    all_stratum = score.stratum("all")
    assert all_stratum.value == pytest.approx(0.0)
    assert all_stratum.n_items == 1


def test_low_iou_prediction_not_matched():
    # Prediction shares no area with GT.
    gt = _gt_from_dict({1: [(Box(x1=0, y1=0, x2=20, y2=40), "person")]})
    preds = (_pred(1, Box(x1=100, y1=100, x2=120, y2=140)),)
    eval_set = _eval_set(
        strata=(StratumLabel(name="all"),),
        item_strata={"1": "all"},
    )
    score = compute_tier3(make_topology(), _pipeline(preds), eval_set, gt)
    assert score.stratum("all").value == 0.0


def test_wrong_class_prediction_not_matched():
    gt = _gt_from_dict({1: [(Box(x1=0, y1=0, x2=20, y2=40), "person")]})
    preds = (_pred(1, Box(x1=0, y1=0, x2=20, y2=40), label="vehicle"),)
    eval_set = _eval_set(
        strata=(StratumLabel(name="all"),),
        item_strata={"1": "all"},
    )
    score = compute_tier3(make_topology(), _pipeline(preds), eval_set, gt)
    assert score.stratum("all").value == 0.0


def test_per_stratum_scores_separate():
    # Image 1 in stratum "easy", image 2 in "hard". Predictions match image 1
    # but not image 2 → easy=1.0, hard=0.0.
    gt = _gt_from_dict({
        1: [(Box(x1=0, y1=0, x2=20, y2=40), "person")],
        2: [(Box(x1=0, y1=0, x2=20, y2=40), "person")],
    })
    preds = (_pred(1, Box(x1=0, y1=0, x2=20, y2=40)),)  # only image 1 predicted
    eval_set = _eval_set(
        strata=(StratumLabel(name="easy"), StratumLabel(name="hard")),
        item_strata={"1": "easy", "2": "hard"},
    )
    score = compute_tier3(make_topology(), _pipeline(preds), eval_set, gt)
    assert score.stratum("easy").value == pytest.approx(1.0)
    assert score.stratum("hard").value == pytest.approx(0.0)


def test_empty_stratum_reported_as_no_estimate():
    # A declared stratum with no assigned items should still appear, with
    # value 0 and n_items 0 — §7.2: never silently drop a stratum.
    gt = _gt_from_dict({1: [(Box(x1=0, y1=0, x2=20, y2=40), "person")]})
    preds = (_pred(1, Box(x1=0, y1=0, x2=20, y2=40)),)
    eval_set = _eval_set(
        strata=(StratumLabel(name="all"), StratumLabel(name="empty_stratum")),
        item_strata={"1": "all"},
    )
    score = compute_tier3(make_topology(), _pipeline(preds), eval_set, gt)
    empty = score.stratum("empty_stratum")
    assert empty.value == 0.0
    assert empty.n_items == 0


# --- Wilson CI ------------------------------------------------------------


def test_wilson_ci_bounds():
    from cv_agent.scoring.tier3 import _wilson_ci

    lo, hi = _wilson_ci(5, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 0.5 < hi


def test_wilson_ci_narrows_with_more_samples():
    from cv_agent.scoring.tier3 import _wilson_ci

    lo_small, hi_small = _wilson_ci(5, 10)
    lo_large, hi_large = _wilson_ci(500, 1000)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_wilson_ci_zero_n():
    from cv_agent.scoring.tier3 import _wilson_ci

    assert _wilson_ci(0, 0) == (0.0, 0.0)


def test_ci_included_in_stratum_score():
    gt = _gt_from_dict({i: [(Box(x1=0, y1=0, x2=20, y2=40), "person")] for i in range(1, 11)})
    preds = tuple(_pred(i, Box(x1=0, y1=0, x2=20, y2=40)) for i in range(1, 6))  # 5/10 match
    eval_set = _eval_set(
        strata=(StratumLabel(name="all"),),
        item_strata={str(i): "all" for i in range(1, 11)},
    )
    score = compute_tier3(make_topology(), _pipeline(preds), eval_set, gt)
    s = score.stratum("all")
    assert s.value == pytest.approx(0.5)
    assert 0.0 < s.ci_low < 0.5 < s.ci_high < 1.0


# --- pooled_with_spread ---------------------------------------------------


def test_pooled_with_spread_returns_triple():
    gt = _gt_from_dict({
        1: [(Box(x1=0, y1=0, x2=20, y2=40), "person")],
        2: [(Box(x1=0, y1=0, x2=20, y2=40), "person")],
    })
    preds = (_pred(1, Box(x1=0, y1=0, x2=20, y2=40)),)
    eval_set = _eval_set(
        strata=(StratumLabel(name="easy"), StratumLabel(name="hard")),
        item_strata={"1": "easy", "2": "hard"},
    )
    score = compute_tier3(make_topology(), _pipeline(preds), eval_set, gt)
    pooled, lo, hi = score.pooled_with_spread()
    # Both strata have n=1, values 1.0 and 0.0; pooled = 0.5, spread [0, 1].
    assert pooled == pytest.approx(0.5)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_pooled_with_spread_empty_strata():
    gt = _gt_from_dict({})
    eval_set = _eval_set(
        strata=(StratumLabel(name="only_declared"),),
        item_strata={"1": "only_declared"},
    )
    # image 1 has no GT → no entry in gt.boxes_by_image → n=0 for stratum.
    score = compute_tier3(make_topology(), _pipeline(()), eval_set, gt)
    assert score.pooled_with_spread() == (0.0, 0.0, 0.0)


# --- compare_tier3 --------------------------------------------------------


def _score_with_values(
    values: dict[str, float], topology=None, eval_name="test_eval"
) -> Tier3Score:
    """Build a Tier3Score directly for comparison testing (bypasses scoring)."""
    from cv_agent.scoring.tier3 import StratumScore

    strata = tuple(
        StratumScore(stratum=n, value=v, ci_low=v, ci_high=v, n_items=10)
        for n, v in values.items()
    )
    return Tier3Score(
        topology=topology or make_topology(),
        metric="recall@0.5",
        scorer_id="recall@0.5",
        eval_set_name=eval_name,
        strata=strata,
    )


def test_compare_tier3_cross_topology_is_legitimate():
    """§17 M3: tier 3 IS meant for cross-topology comparison."""
    a = _score_with_values({"easy": 0.9, "hard": 0.5}, topology=make_topology())
    b = _score_with_values(
        {"easy": 0.8, "hard": 0.6},
        topology=make_topology(detector=make_detector(name="other")),
    )
    # a beats b on easy, b beats a on hard.
    assert compare_tier3(a, b, on_stratum="easy") == 1
    assert compare_tier3(a, b, on_stratum="hard") == -1


def test_compare_tier3_scorer_mismatch_raises():
    from cv_agent.scoring.tier3 import Tier3Score, StratumScore

    a = _score_with_values({"all": 0.5})
    b_bad = Tier3Score(
        topology=make_topology(),
        metric="fake",
        scorer_id="different_scorer",
        eval_set_name="test_eval",
        strata=(StratumScore(stratum="all", value=0.6, ci_low=0.5, ci_high=0.7, n_items=10),),
    )
    with pytest.raises(ValueError, match="scorer mismatch"):
        compare_tier3(a, b_bad, on_stratum="all")


def test_compare_tier3_eval_set_mismatch_raises():
    a = _score_with_values({"all": 0.5}, eval_name="eval_a")
    b = _score_with_values({"all": 0.6}, eval_name="eval_b")
    with pytest.raises(ValueError, match="eval-set mismatch"):
        compare_tier3(a, b, on_stratum="all")


# --- dominance ------------------------------------------------------------


def test_dominance_strict():
    a = _score_with_values({"easy": 0.9, "hard": 0.7})
    b = _score_with_values({"easy": 0.8, "hard": 0.5})
    # a strictly greater on both strata.
    assert dominates(a, b)
    assert not dominates(b, a)


def test_no_dominance_when_traded_off():
    a = _score_with_values({"easy": 0.9, "hard": 0.5})
    b = _score_with_values({"easy": 0.8, "hard": 0.6})
    # a better on easy, b better on hard — Pareto frontier.
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_equal_scores_do_not_dominate():
    a = _score_with_values({"easy": 0.9, "hard": 0.7})
    b = _score_with_values({"easy": 0.9, "hard": 0.7})
    # a is >= b everywhere but strict-greater nowhere → no dominance.
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominance_across_different_eval_sets_is_false():
    a = _score_with_values({"all": 0.9}, eval_name="A")
    b = _score_with_values({"all": 0.1}, eval_name="B")
    # Different eval sets → not comparable.
    assert not dominates(a, b)
