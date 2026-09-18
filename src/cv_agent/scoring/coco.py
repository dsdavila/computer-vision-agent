"""Minimal COCO ground-truth loader for tier-3 (§6, §17 M3).

Parses the essentials — images + annotations + categories — and builds a
per-image bucket of GT boxes with class labels. Full COCO semantics
(iscrowd, area ranges, per-image licenses, keypoints) are out of scope;
this is what tier-3 needs to compute matching-based metrics like
recall@IoU.

If we later need pycocotools' evaluation semantics (interpolated AP,
area buckets), swap this out — the tier-3 scorer contract is
`(pipeline_output, gt, eval_set) -> per-stratum scores`, so the loader
is replaceable without disturbing the tier-3 API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cv_agent.profiler.proposals import Box


@dataclass(frozen=True)
class GTBox:
    """One ground-truth annotation: an axis-aligned box and its class label."""

    box: Box
    label: str


@dataclass(frozen=True)
class CocoGroundTruth:
    """Parsed COCO GT indexed by image_id. Includes the id→name category
    mapping so scorers can filter by class label."""

    boxes_by_image: dict[int, tuple[GTBox, ...]]
    categories: dict[int, str]


def load_coco(path: str | Path) -> CocoGroundTruth:
    """Load a COCO-format JSON file. Expects the standard
    {images, annotations, categories} keys."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    categories: dict[int, str] = {
        int(c["id"]): c["name"] for c in payload.get("categories", [])
    }

    buckets: dict[int, list[GTBox]] = {}
    for ann in payload.get("annotations", []):
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            # Degenerate annotation — skip rather than propagate the crash
            # from Box's constructor. Real COCO files sometimes carry these.
            continue
        gt = GTBox(
            box=Box(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h)),
            label=categories.get(category_id, f"category_{category_id}"),
        )
        buckets.setdefault(image_id, []).append(gt)

    boxes_by_image = {k: tuple(v) for k, v in buckets.items()}
    return CocoGroundTruth(boxes_by_image=boxes_by_image, categories=categories)
