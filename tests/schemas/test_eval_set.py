"""EvalSet + DiagnosticSet schema tests."""

import pytest
from pydantic import ValidationError

from cv_agent.schemas import (
    DiagnosticSet,
    EvalSet,
    GroundTruthFile,
    GroundTruthFormat,
    StratumLabel,
)


def _gt_coco() -> GroundTruthFile:
    return GroundTruthFile(
        path="eval/dock_v0/detection_gt.json",
        format=GroundTruthFormat.COCO,
        sha256="a" * 64,
    )


def _gt_mot() -> GroundTruthFile:
    return GroundTruthFile(
        path="eval/dock_v0/tracking_gt/",
        format=GroundTruthFormat.MOT,
    )


def _strata() -> tuple[StratumLabel, ...]:
    return (
        StratumLabel(name="low_pixels_high_congestion"),
        StratumLabel(name="low_pixels_low_congestion"),
        StratumLabel(name="high_pixels_high_congestion"),
        StratumLabel(name="high_pixels_low_congestion"),
        StratumLabel(name="low_light"),
        StratumLabel(name="high_motion"),
    )


def _valid_eval_set() -> EvalSet:
    return EvalSet(
        name="dock_v0_detection_eval",
        ground_truth=_gt_coco(),
        strata=_strata(),
        item_strata={
            "1": "low_pixels_high_congestion",
            "2": "low_pixels_low_congestion",
            "3": "high_pixels_high_congestion",
            "4": "high_pixels_low_congestion",
            "5": "low_light",
            "6": "high_motion",
        },
        scorer_id="coco_map_v1",
        catalog_version="v0-2026-09-17",
        task_contract_name="dock_surveillance_v0",
    )


def _valid_diagnostic_set() -> DiagnosticSet:
    return DiagnosticSet(
        name="dock_v0_diagnostic",
        ground_truth=_gt_coco(),
        catalog_version="v0-2026-09-17",
        task_contract_name="dock_surveillance_v0",
        compared_configs=("cfg-a", "cfg-b", "cfg-c"),
    )


# --- happy path -------------------------------------------------------------


def test_valid_eval_set_constructs():
    e = _valid_eval_set()
    assert e.ground_truth.format is GroundTruthFormat.COCO
    assert len(e.strata) == 6
    assert e.item_strata["1"] == "low_pixels_high_congestion"


def test_valid_diagnostic_set_constructs():
    d = _valid_diagnostic_set()
    assert d.compared_configs == ("cfg-a", "cfg-b", "cfg-c")
    assert d.ground_truth.format is GroundTruthFormat.COCO


def test_diagnostic_set_empty_compared_configs_ok():
    # v0 pre-M3 state: no candidate configs yet.
    d = DiagnosticSet(
        name="empty",
        ground_truth=_gt_coco(),
        catalog_version="v0",
        task_contract_name="c",
    )
    assert d.compared_configs == ()


def test_mot_format_supported():
    gt = _gt_mot()
    assert gt.format is GroundTruthFormat.MOT
    assert gt.sha256 is None


# --- ground-truth-file invariants -------------------------------------------


def test_sha256_optional():
    gt = GroundTruthFile(path="x.json", format=GroundTruthFormat.COCO)
    assert gt.sha256 is None


def test_sha256_must_be_64_lowercase_hex():
    with pytest.raises(ValueError):
        GroundTruthFile(
            path="x.json",
            format=GroundTruthFormat.COCO,
            sha256="not-a-hash",
        )


def test_sha256_uppercase_rejected():
    with pytest.raises(ValueError):
        GroundTruthFile(
            path="x.json",
            format=GroundTruthFormat.COCO,
            sha256="A" * 64,
        )


def test_ground_truth_format_enum_only():
    with pytest.raises(ValidationError):
        GroundTruthFile.model_validate(
            {"path": "x.json", "format": "yolo"}
        )


# --- eval-set cross-field invariants ---------------------------------------


def test_item_strata_value_must_reference_declared_stratum():
    with pytest.raises(ValueError):
        EvalSet(
            name="bad",
            ground_truth=_gt_coco(),
            strata=_strata(),
            item_strata={"1": "not_a_declared_stratum"},
            scorer_id="coco_map_v1",
            catalog_version="v0",
            task_contract_name="c",
        )


def test_stratum_names_must_be_unique():
    with pytest.raises(ValueError):
        EvalSet(
            name="bad",
            ground_truth=_gt_coco(),
            strata=(
                StratumLabel(name="x"),
                StratumLabel(name="x"),
            ),
            item_strata={"1": "x"},
            scorer_id="coco_map_v1",
            catalog_version="v0",
            task_contract_name="c",
        )


def test_empty_item_strata_rejected():
    with pytest.raises(ValueError):
        EvalSet(
            name="bad",
            ground_truth=_gt_coco(),
            strata=_strata(),
            item_strata={},
            scorer_id="coco_map_v1",
            catalog_version="v0",
            task_contract_name="c",
        )


def test_strata_required_non_empty():
    with pytest.raises(ValidationError):
        EvalSet(
            name="bad",
            ground_truth=_gt_coco(),
            strata=(),
            item_strata={"1": "x"},
            scorer_id="coco_map_v1",
            catalog_version="v0",
            task_contract_name="c",
        )


# --- required top-level fields ---------------------------------------------


def test_eval_set_empty_scorer_id_rejected():
    with pytest.raises(ValidationError):
        EvalSet.model_validate(
            _valid_eval_set().model_dump() | {"scorer_id": ""}
        )


def test_eval_set_empty_catalog_version_rejected():
    with pytest.raises(ValidationError):
        EvalSet.model_validate(
            _valid_eval_set().model_dump() | {"catalog_version": ""}
        )


def test_eval_set_empty_task_contract_name_rejected():
    with pytest.raises(ValidationError):
        EvalSet.model_validate(
            _valid_eval_set().model_dump() | {"task_contract_name": ""}
        )


# --- schema-drift guards ----------------------------------------------------


def test_frozen_eval_set():
    e = _valid_eval_set()
    with pytest.raises(ValidationError):
        e.name = "renamed"  # type: ignore[misc]


def test_frozen_diagnostic_set():
    d = _valid_diagnostic_set()
    with pytest.raises(ValidationError):
        d.name = "renamed"  # type: ignore[misc]


def test_extra_field_forbidden_on_eval_set():
    with pytest.raises(ValidationError):
        EvalSet.model_validate(
            _valid_eval_set().model_dump() | {"undocumented": 1}
        )


def test_extra_field_forbidden_on_diagnostic_set():
    with pytest.raises(ValidationError):
        DiagnosticSet.model_validate(
            _valid_diagnostic_set().model_dump() | {"undocumented": 1}
        )
