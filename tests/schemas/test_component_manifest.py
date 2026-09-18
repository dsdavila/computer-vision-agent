"""ComponentManifest schema tests — exercise the shape and its invariants."""

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from cv_agent.schemas import (
    BoolParam,
    CategoricalParam,
    ComponentKind,
    ComponentManifest,
    CostMeasurement,
    FloatParam,
    HardwareEnvelope,
    IOContract,
    IOKind,
    IntParam,
    ParamSpec,
    Preconditions,
    ReferenceCost,
    TargetCostMeasurement,
)


def _reference_gpu() -> HardwareEnvelope:
    return HardwareEnvelope(gpu_model="NVIDIA A100 80GB", vram_gb=80, cpu_cores=32)


def _target_gpu() -> HardwareEnvelope:
    return HardwareEnvelope(gpu_model="NVIDIA L4 24GB", vram_gb=24, cpu_cores=16)


def _valid_reference_cost() -> ReferenceCost:
    return ReferenceCost(
        reference_hardware=_reference_gpu(),
        scaling_function="linear_flops",
        measurements=(
            CostMeasurement(
                config={"input_res": 640, "tile_count": 1, "confidence_threshold": 0.4},
                latency_ms=18.0,
                vram_mb=1400.0,
            ),
            CostMeasurement(
                config={"input_res": 640, "tile_count": 4, "confidence_threshold": 0.4},
                latency_ms=82.0,
                vram_mb=1800.0,
            ),
        ),
    )


def _valid_manifest() -> ComponentManifest:
    return ComponentManifest(
        name="general_closed_set_detector_v1",
        kind=ComponentKind.DETECTOR,
        catalog_version="v0-2026-09-17",
        license="Apache-2.0",
        preconditions=Preconditions(
            min_input_width_px=320,
            min_input_height_px=320,
            requires_gpu=True,
            min_vram_mb=1024,
            supported_classes=("person", "vehicle"),
        ),
        io_contract=IOContract(
            consumes=(IOKind.FRAME,),
            produces=(IOKind.BOXES,),
        ),
        parameters=(
            FloatParam(name="confidence_threshold", min=0.0, max=1.0, default=0.4),
            IntParam(name="input_res", min=320, max=1536, default=640),
            IntParam(name="tile_count", min=1, max=16, default=1),
            BoolParam(name="fp16", default=True),
            CategoricalParam(
                name="nms_kind",
                values=("standard", "soft", "diou"),
                default="standard",
            ),
        ),
        reference_cost=_valid_reference_cost(),
    )


# --- happy path -------------------------------------------------------------


def test_valid_manifest_constructs():
    m = _valid_manifest()
    assert m.kind is ComponentKind.DETECTOR
    assert m.parameters[0].kind == "float"
    assert m.reference_cost.measurements[1].latency_ms == 82.0


def test_target_cost_measurement_constructs():
    tc = TargetCostMeasurement(
        component_name="general_closed_set_detector_v1",
        catalog_version="v0-2026-09-17",
        hardware=_target_gpu(),
        config={"input_res": 640, "tile_count": 4, "confidence_threshold": 0.4},
        latency_ms=210.0,
        vram_mb=3400.0,
        measured_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
    )
    assert tc.latency_ms == 210.0
    assert tc.hardware.gpu_model == "NVIDIA L4 24GB"


# --- schema-drift guards ----------------------------------------------------


def test_frozen():
    m = _valid_manifest()
    with pytest.raises(ValidationError):
        m.name = "changed"  # type: ignore[misc]


def test_extra_field_forbidden_on_manifest():
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate(
            _valid_manifest().model_dump() | {"undocumented_field": 1}
        )


def test_extra_field_forbidden_on_reference_cost():
    with pytest.raises(ValidationError):
        ReferenceCost.model_validate(
            _valid_reference_cost().model_dump() | {"junk": 1}
        )


# --- discriminated union on parameters --------------------------------------


def test_param_kinds_round_trip():
    adapter = TypeAdapter(ParamSpec)
    for param in _valid_manifest().parameters:
        dumped = adapter.dump_python(param)
        assert "kind" in dumped
        rebuilt = adapter.validate_python(dumped)
        assert type(rebuilt) is type(param)


def test_missing_param_discriminator_fails():
    adapter = TypeAdapter(ParamSpec)
    with pytest.raises(ValidationError):
        adapter.validate_python({"name": "x", "min": 0.0, "max": 1.0, "default": 0.5})


def test_float_param_default_within_bounds():
    with pytest.raises(ValueError):
        FloatParam(name="thr", min=0.0, max=1.0, default=1.5)


def test_int_param_min_greater_than_max_rejected():
    with pytest.raises(ValueError):
        IntParam(name="res", min=100, max=10, default=50)


def test_categorical_default_must_be_in_values():
    with pytest.raises(ValueError):
        CategoricalParam(name="nms", values=("a", "b"), default="c")


# --- cost invariants --------------------------------------------------------


def test_reference_cost_requires_at_least_one_measurement():
    with pytest.raises(ValidationError):
        ReferenceCost(
            reference_hardware=_reference_gpu(),
            scaling_function="linear_flops",
            measurements=(),
        )


def test_negative_latency_rejected():
    with pytest.raises(ValidationError):
        CostMeasurement(
            config={"input_res": 640},
            latency_ms=-1.0,
            vram_mb=100.0,
        )


def test_negative_vram_rejected():
    with pytest.raises(ValidationError):
        CostMeasurement(
            config={"input_res": 640},
            latency_ms=10.0,
            vram_mb=-1.0,
        )


# --- required top-level fields ---------------------------------------------


def test_missing_catalog_version_fails():
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate(
            _valid_manifest().model_dump() | {"catalog_version": ""}
        )


def test_missing_license_fails():
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate(
            _valid_manifest().model_dump() | {"license": ""}
        )


def test_missing_reference_cost_fails():
    data = _valid_manifest().model_dump()
    del data["reference_cost"]
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate(data)


# --- enum-only values -------------------------------------------------------


def test_component_kind_enum_only():
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate(
            _valid_manifest().model_dump() | {"kind": "auxiliary"}
        )


def test_io_kind_enum_only():
    with pytest.raises(ValidationError):
        IOContract.model_validate({"consumes": ["frame"], "produces": ["nonsense"]})
