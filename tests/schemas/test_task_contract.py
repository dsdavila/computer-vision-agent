"""TaskContract schema tests — exercise the shape, coordinate bounds, and
cross-field invariants (predicate targets vs. ontology, unique predicate
names, success-criterion classes, etc.)."""

import pytest
from pydantic import ValidationError

from cv_agent.schemas import (
    DeploymentHardware,
    DwellTimePredicate,
    HardwareEnvelope,
    LineCrossingDirection,
    LineCrossingPredicate,
    Ontology,
    OperatingBias,
    OperatingPoint,
    ROIPredicate,
    ScoringMetric,
    SuccessCriterion,
    TaskContract,
)


def _deployment_hw() -> DeploymentHardware:
    return DeploymentHardware(
        hardware=HardwareEnvelope(gpu_model="NVIDIA L4 24GB", vram_gb=24, cpu_cores=16),
        latency_budget_ms=100.0,
        min_throughput_fps=15.0,
    )


def _ontology() -> Ontology:
    return Ontology(
        closed_set_classes=("person", "vehicle"),
        open_vocab_descriptions=("delivery truck",),
    )


def _roi() -> ROIPredicate:
    return ROIPredicate(
        name="loading_dock",
        polygon_normalized=((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)),
        target_classes=("vehicle", "delivery truck"),
    )


def _line() -> LineCrossingPredicate:
    return LineCrossingPredicate(
        name="entry_gate",
        line_start_normalized=(0.0, 0.5),
        line_end_normalized=(1.0, 0.5),
        direction=LineCrossingDirection.START_TO_END,
        target_classes=("person",),
    )


def _dwell() -> DwellTimePredicate:
    return DwellTimePredicate(
        name="lingering_at_dock",
        polygon_normalized=((0.2, 0.3), (0.8, 0.3), (0.8, 0.7), (0.2, 0.7)),
        target_classes=("person",),
        min_seconds=30.0,
    )


def _operating_point() -> OperatingPoint:
    return OperatingPoint(
        bias=OperatingBias.RECALL_WEIGHTED,
        target_recall=0.85,
        target_precision=None,
    )


def _success() -> tuple[SuccessCriterion, ...]:
    return (
        SuccessCriterion(metric=ScoringMetric.RECALL, per_class="person", min_value=0.85),
        SuccessCriterion(metric=ScoringMetric.FALSE_ALARM_RATE, per_class=None, max_value=0.02),
    )


def _valid_contract() -> TaskContract:
    return TaskContract(
        name="dock_surveillance_v0",
        ontology=_ontology(),
        predicates=(_roi(), _line(), _dwell()),
        operating_point=_operating_point(),
        deployment_hardware=_deployment_hw(),
        success_criteria=_success(),
        catalog_version="v0-2026-09-17",
    )


# --- happy path -------------------------------------------------------------


def test_valid_contract_constructs():
    c = _valid_contract()
    assert c.name == "dock_surveillance_v0"
    assert len(c.predicates) == 3
    assert c.ontology.all_target_names == ("person", "vehicle", "delivery truck")


# --- ontology invariants ----------------------------------------------------


def test_empty_ontology_rejected():
    with pytest.raises(ValueError):
        Ontology(closed_set_classes=(), open_vocab_descriptions=())


def test_duplicate_closed_set_rejected():
    with pytest.raises(ValueError):
        Ontology(closed_set_classes=("person", "person"))


def test_empty_class_name_rejected():
    with pytest.raises(ValueError):
        Ontology(closed_set_classes=("person", ""))


# --- predicate coordinate bounds --------------------------------------------


def test_roi_polygon_needs_three_points():
    with pytest.raises(ValueError):
        ROIPredicate(
            name="tiny",
            polygon_normalized=((0.1, 0.2), (0.3, 0.4)),
            target_classes=("person",),
        )


def test_roi_polygon_out_of_bounds_rejected():
    with pytest.raises(ValueError):
        ROIPredicate(
            name="oob",
            polygon_normalized=((0.1, 0.2), (1.5, 0.2), (0.5, 0.5)),
            target_classes=("person",),
        )


def test_line_endpoints_must_differ():
    with pytest.raises(ValueError):
        LineCrossingPredicate(
            name="degenerate",
            line_start_normalized=(0.5, 0.5),
            line_end_normalized=(0.5, 0.5),
            direction=LineCrossingDirection.EITHER,
            target_classes=("person",),
        )


def test_dwell_time_must_be_positive():
    with pytest.raises(ValidationError):
        DwellTimePredicate(
            name="zero_dwell",
            polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
            target_classes=("person",),
            min_seconds=0.0,
        )


def test_predicate_target_classes_required():
    with pytest.raises(ValidationError):
        ROIPredicate(
            name="empty_targets",
            polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
            target_classes=(),
        )


# --- cross-field invariants -------------------------------------------------


def test_predicate_targets_must_be_in_ontology():
    with pytest.raises(ValueError):
        TaskContract(
            name="bad",
            ontology=Ontology(closed_set_classes=("person",)),
            predicates=(
                ROIPredicate(
                    name="area",
                    polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                    target_classes=("bicycle",),  # not in ontology
                ),
            ),
            operating_point=_operating_point(),
            deployment_hardware=_deployment_hw(),
            success_criteria=_success(),
            catalog_version="v0",
        )


def test_success_criterion_per_class_must_be_in_ontology():
    with pytest.raises(ValueError):
        TaskContract(
            name="bad",
            ontology=Ontology(closed_set_classes=("person",)),
            predicates=(_line(),) if False else (  # noqa: SIM108 - keep contract self-contained
                ROIPredicate(
                    name="area",
                    polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                    target_classes=("person",),
                ),
            ),
            operating_point=_operating_point(),
            deployment_hardware=_deployment_hw(),
            success_criteria=(
                SuccessCriterion(
                    metric=ScoringMetric.RECALL,
                    per_class="vehicle",  # not in ontology
                    min_value=0.8,
                ),
            ),
            catalog_version="v0",
        )


def test_predicate_names_must_be_unique():
    dup_line = LineCrossingPredicate(
        name="entry_gate",  # collides with _line().name
        line_start_normalized=(0.0, 0.3),
        line_end_normalized=(1.0, 0.3),
        direction=LineCrossingDirection.EITHER,
        target_classes=("person",),
    )
    with pytest.raises(ValueError):
        TaskContract(
            name="dup_predicates",
            ontology=_ontology(),
            predicates=(_line(), dup_line),
            operating_point=_operating_point(),
            deployment_hardware=_deployment_hw(),
            success_criteria=_success(),
            catalog_version="v0",
        )


def test_open_vocab_target_is_a_valid_predicate_target():
    # A predicate can name an open-vocab description; the ontology check
    # accepts it because open-vocab descriptions are part of all_target_names.
    contract = TaskContract(
        name="open_vocab_ok",
        ontology=Ontology(
            closed_set_classes=(),
            open_vocab_descriptions=("delivery truck",),
        ),
        predicates=(
            ROIPredicate(
                name="dock",
                polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                target_classes=("delivery truck",),
            ),
        ),
        operating_point=OperatingPoint(bias=OperatingBias.BALANCED),
        deployment_hardware=_deployment_hw(),
        success_criteria=(
            SuccessCriterion(
                metric=ScoringMetric.RECALL,
                per_class="delivery truck",
                min_value=0.7,
            ),
        ),
        catalog_version="v0",
    )
    assert contract.predicates[0].target_classes == ("delivery truck",)


# --- success criterion invariants -------------------------------------------


def test_success_criterion_needs_at_least_one_bound():
    with pytest.raises(ValueError):
        SuccessCriterion(metric=ScoringMetric.RECALL, per_class=None)


def test_success_criterion_min_greater_than_max_rejected():
    with pytest.raises(ValueError):
        SuccessCriterion(
            metric=ScoringMetric.RECALL,
            per_class=None,
            min_value=0.9,
            max_value=0.5,
        )


def test_success_criterion_value_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SuccessCriterion(metric=ScoringMetric.RECALL, per_class=None, min_value=1.5)


# --- operating-point / hardware invariants ----------------------------------


def test_operating_point_probabilities_bounded():
    with pytest.raises(ValidationError):
        OperatingPoint(bias=OperatingBias.RECALL_WEIGHTED, target_recall=1.5)


def test_latency_budget_must_be_positive():
    with pytest.raises(ValidationError):
        DeploymentHardware(
            hardware=HardwareEnvelope(gpu_model="X", vram_gb=1, cpu_cores=1),
            latency_budget_ms=0.0,
        )


# --- schema-drift guards ----------------------------------------------------


def test_frozen():
    c = _valid_contract()
    with pytest.raises(ValidationError):
        c.name = "renamed"  # type: ignore[misc]


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        TaskContract.model_validate(
            _valid_contract().model_dump() | {"undocumented_field": 1}
        )


def test_predicate_discriminator_required():
    with pytest.raises(ValidationError):
        TaskContract.model_validate(
            _valid_contract().model_dump()
            | {
                "predicates": [
                    {
                        # missing "kind"
                        "name": "x",
                        "polygon_normalized": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
                        "target_classes": ["person"],
                    }
                ]
            }
        )
