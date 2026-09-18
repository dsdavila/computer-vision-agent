"""Shared fixtures for scoring tests. Builds ComponentManifests + Topology +
TaskContract with sensible defaults so individual tests override only what
they need."""

from __future__ import annotations

from datetime import datetime, timezone

from cv_agent.schemas import (
    ComponentKind,
    ComponentManifest,
    CostMeasurement,
    DeploymentHardware,
    HardwareEnvelope,
    IOContract,
    IOKind,
    Ontology,
    OperatingBias,
    OperatingPoint,
    Preconditions,
    ROIPredicate,
    ReferenceCost,
    ScoringMetric,
    SuccessCriterion,
    TargetCostMeasurement,
    TaskContract,
)
from cv_agent.scoring.topology import Topology


CATALOG = "v0-2026-09-17"


def make_detector(
    *,
    name: str = "general_v1",
    license: str = "Apache-2.0",
    supported_classes: tuple[str, ...] | None = ("person", "vehicle"),
) -> ComponentManifest:
    return ComponentManifest(
        name=name,
        kind=ComponentKind.DETECTOR,
        catalog_version=CATALOG,
        license=license,
        preconditions=Preconditions(
            requires_gpu=True,
            min_vram_mb=1024,
            supported_classes=supported_classes,
        ),
        io_contract=IOContract(consumes=(IOKind.FRAME,), produces=(IOKind.BOXES,)),
        reference_cost=_ref_cost(),
    )


def make_tile(*, name: str = "tile_off_v1", license: str = "Apache-2.0") -> ComponentManifest:
    return ComponentManifest(
        name=name,
        kind=ComponentKind.TILE,
        catalog_version=CATALOG,
        license=license,
        preconditions=Preconditions(),
        io_contract=IOContract(consumes=(IOKind.FRAME,), produces=(IOKind.FRAME_TILES,)),
        reference_cost=_ref_cost(),
    )


def make_tracker(*, name: str = "motion_only_v1", license: str = "Apache-2.0") -> ComponentManifest:
    return ComponentManifest(
        name=name,
        kind=ComponentKind.TRACKER,
        catalog_version=CATALOG,
        license=license,
        preconditions=Preconditions(),
        io_contract=IOContract(consumes=(IOKind.BOXES,), produces=(IOKind.TRACKS,)),
        reference_cost=_ref_cost(),
    )


def make_reid(*, name: str = "reid_off_v1", license: str = "Apache-2.0") -> ComponentManifest:
    return ComponentManifest(
        name=name,
        kind=ComponentKind.REID,
        catalog_version=CATALOG,
        license=license,
        preconditions=Preconditions(),
        io_contract=IOContract(consumes=(IOKind.CROPS,), produces=(IOKind.EMBEDDINGS,)),
        reference_cost=_ref_cost(),
    )


def _ref_cost() -> ReferenceCost:
    return ReferenceCost(
        reference_hardware=HardwareEnvelope(
            gpu_model="NVIDIA A100 80GB", vram_gb=80, cpu_cores=32
        ),
        scaling_function="linear_flops",
        measurements=(
            CostMeasurement(
                config={"input_res": 640},
                latency_ms=10.0,
                vram_mb=1024.0,
            ),
        ),
    )


def make_topology(
    *,
    detector: ComponentManifest | None = None,
    tile: ComponentManifest | None = None,
    tracker: ComponentManifest | None = None,
    reid: ComponentManifest | None = None,
) -> Topology:
    return Topology(
        detector=detector or make_detector(),
        tile=tile or make_tile(),
        tracker=tracker or make_tracker(),
        reid=reid or make_reid(),
    )


def make_target_costs(
    latency_by_component: dict[str, float] | None = None,
    vram_by_component: dict[str, float] | None = None,
) -> dict[str, TargetCostMeasurement]:
    """Build target-cost measurements for the four v0 components. Any
    per-component overrides in latency_by_component / vram_by_component
    replace the defaults."""
    latency = latency_by_component or {}
    vram = vram_by_component or {}
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    hw = HardwareEnvelope(gpu_model="L4", vram_gb=24, cpu_cores=16)
    out = {}
    for name in ("general_v1", "tile_off_v1", "motion_only_v1", "reid_off_v1"):
        out[name] = TargetCostMeasurement(
            component_name=name,
            catalog_version=CATALOG,
            hardware=hw,
            config={},
            latency_ms=latency.get(name, 15.0),
            vram_mb=vram.get(name, 1200.0),
            measured_at=now,
        )
    return out


def make_task_contract(
    *,
    ontology: Ontology | None = None,
    latency_budget_ms: float = 100.0,
    vram_gb: int = 24,
    catalog_version: str = CATALOG,
) -> TaskContract:
    return TaskContract(
        name="test",
        ontology=ontology or Ontology(closed_set_classes=("person", "vehicle")),
        predicates=(
            ROIPredicate(
                name="area",
                polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                target_classes=("person",),
            ),
        ),
        operating_point=OperatingPoint(bias=OperatingBias.BALANCED),
        deployment_hardware=DeploymentHardware(
            hardware=HardwareEnvelope(gpu_model="L4", vram_gb=vram_gb, cpu_cores=16),
            latency_budget_ms=latency_budget_ms,
        ),
        success_criteria=(
            SuccessCriterion(
                metric=ScoringMetric.RECALL,
                per_class="person",
                min_value=0.8,
            ),
        ),
        catalog_version=catalog_version,
    )
