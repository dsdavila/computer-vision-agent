"""ComponentManifest — one entry in the capability registry (§3, §13.1, §16 #3).

Cost is a **function of config**, not a scalar (§13.1). Tiling a general detector
at 2x2 with overlap costs ~4-5x an untiled pass; scalar cost teaches the case
base that tiling is free (§14 risk row). Two tiers are frozen here:

- ReferenceCost lives on the manifest, measured on a canonical reference SKU
  at registry-register time. Used by the planner during search with a declared
  hardware scaling function.
- TargetCostMeasurement is a separate per-deployment record produced by the M4
  calibration pass and consumed by the tier-0 gate. It references the component
  by name + catalog_version, not by embedded copy.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


_frozen = ConfigDict(frozen=True, extra="forbid")


# --- component kind ---------------------------------------------------------

class ComponentKind(str, Enum):
    """The four v0 catalog axes (§15)."""

    DETECTOR = "detector"
    TILE = "tile"
    TRACKER = "tracker"
    REID = "reid"


# --- I/O contract -----------------------------------------------------------

class IOKind(str, Enum):
    """Kinds a component consumes or produces. Kind-level only — real tensor
    shapes and dtypes are M2 runtime concerns, not manifest concerns."""

    FRAME = "frame"
    FRAME_TILES = "frame_tiles"
    BOXES = "boxes"
    BOXES_PER_TILE = "boxes_per_tile"
    TRACKS = "tracks"
    CROPS = "crops"
    EMBEDDINGS = "embeddings"


class IOContract(BaseModel):
    model_config = _frozen

    consumes: tuple[IOKind, ...]
    produces: tuple[IOKind, ...]


# --- preconditions ----------------------------------------------------------

class Preconditions(BaseModel):
    """Declared preconditions for running the component (§3). All fields optional;
    a component that imposes no constraint on a dimension leaves it None."""

    model_config = _frozen

    min_input_width_px: int | None = Field(default=None, ge=1)
    min_input_height_px: int | None = Field(default=None, ge=1)
    requires_gpu: bool = False
    min_vram_mb: int | None = Field(default=None, ge=0)
    min_frame_rate_hz: float | None = Field(default=None, ge=0)
    # None = open-vocabulary; a concrete list = closed-set with that vocabulary.
    supported_classes: tuple[str, ...] | None = None


# --- parameter space --------------------------------------------------------
#
# Discriminated union on `kind` so the planner and staged-search code can walk
# the parameter space without introspection tricks.

class FloatParam(BaseModel):
    model_config = _frozen

    kind: Literal["float"] = "float"
    name: str = Field(min_length=1)
    min: float
    max: float
    default: float

    def model_post_init(self, _context) -> None:  # noqa: D401
        if self.min > self.max:
            raise ValueError(f"FloatParam {self.name!r}: min > max")
        if not (self.min <= self.default <= self.max):
            raise ValueError(f"FloatParam {self.name!r}: default outside [min, max]")


class IntParam(BaseModel):
    model_config = _frozen

    kind: Literal["int"] = "int"
    name: str = Field(min_length=1)
    min: int
    max: int
    default: int

    def model_post_init(self, _context) -> None:
        if self.min > self.max:
            raise ValueError(f"IntParam {self.name!r}: min > max")
        if not (self.min <= self.default <= self.max):
            raise ValueError(f"IntParam {self.name!r}: default outside [min, max]")


class BoolParam(BaseModel):
    model_config = _frozen

    kind: Literal["bool"] = "bool"
    name: str = Field(min_length=1)
    default: bool


class CategoricalParam(BaseModel):
    model_config = _frozen

    kind: Literal["categorical"] = "categorical"
    name: str = Field(min_length=1)
    values: tuple[str, ...] = Field(min_length=1)
    default: str

    def model_post_init(self, _context) -> None:
        if self.default not in self.values:
            raise ValueError(
                f"CategoricalParam {self.name!r}: default {self.default!r} not in values"
            )


ParamSpec = Annotated[
    Union[FloatParam, IntParam, BoolParam, CategoricalParam],
    Field(discriminator="kind"),
]


# --- cost model (§13.1) -----------------------------------------------------

class HardwareEnvelope(BaseModel):
    """A hardware profile the manifest or a target measurement is pinned to."""

    model_config = _frozen

    gpu_model: str = Field(min_length=1)
    vram_gb: int = Field(ge=1)
    cpu_cores: int = Field(ge=1)


# Values a config parameter can take. Matches the ParamSpec discriminated union
# at the value level, kept loose here so a config can carry any parameter type.
ConfigValue = Union[float, int, bool, str]


class CostMeasurement(BaseModel):
    """One measured point in the (config -> cost) function. A component's
    reference cost is a collection of these across the parameter space."""

    model_config = _frozen

    config: dict[str, ConfigValue]
    latency_ms: float = Field(ge=0)
    vram_mb: float = Field(ge=0)


class ReferenceCost(BaseModel):
    """§13.1 tier-1 cost, on-manifest. Data points, not a formula — the formula
    lives in the scaling function named here, implemented in M4 calibration."""

    model_config = _frozen

    reference_hardware: HardwareEnvelope
    scaling_function: str = Field(min_length=1)
    measurements: tuple[CostMeasurement, ...] = Field(min_length=1)


class TargetCostMeasurement(BaseModel):
    """§13.1 tier-2 cost, off-manifest. Produced by the M4 calibration pass on
    the hardware named in the TaskContract's envelope; consumed by the tier-0
    gate (§6). Ledger entries carry a reference to a measurement of this shape
    as `cost_measured` (§9)."""

    model_config = _frozen

    component_name: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    hardware: HardwareEnvelope
    config: dict[str, ConfigValue]
    latency_ms: float = Field(ge=0)
    vram_mb: float = Field(ge=0)
    measured_at: datetime


# --- the manifest -----------------------------------------------------------

class ComponentManifest(BaseModel):
    """One catalog entry (§3 capability registry). All catalog components — the
    24 topologies in §15 are built from Cartesian products of these — carry a
    manifest of this shape."""

    model_config = _frozen

    name: str = Field(min_length=1)
    kind: ComponentKind
    catalog_version: str = Field(min_length=1)
    license: str = Field(min_length=1)
    preconditions: Preconditions
    io_contract: IOContract
    parameters: tuple[ParamSpec, ...] = ()
    reference_cost: ReferenceCost
