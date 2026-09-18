"""TaskContract — spec compiler output (§3, §16 #1).

Human-confirmed interface. Everything downstream compiles against it. Frozen
once created; a new run against a different catalog is a new contract or a
migration (§13), not an in-place edit.
"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from cv_agent.schemas.component_manifest import HardwareEnvelope

_frozen = ConfigDict(frozen=True, extra="forbid")


# --- ontology ---------------------------------------------------------------


class Ontology(BaseModel):
    """Targets to detect / track. Closed-set names come from the catalog's
    supported vocabulary; open-vocab descriptions are free-form natural
    language (e.g. "delivery truck"). At least one target is required."""

    model_config = _frozen

    closed_set_classes: tuple[str, ...] = ()
    open_vocab_descriptions: tuple[str, ...] = ()

    def model_post_init(self, _context) -> None:
        if not self.closed_set_classes and not self.open_vocab_descriptions:
            raise ValueError("Ontology must include at least one closed-set class or open-vocab description")
        if len(set(self.closed_set_classes)) != len(self.closed_set_classes):
            raise ValueError("Ontology closed_set_classes contains duplicates")
        for name in self.closed_set_classes:
            if not name:
                raise ValueError("Ontology closed_set_classes contains empty string")
        for desc in self.open_vocab_descriptions:
            if not desc:
                raise ValueError("Ontology open_vocab_descriptions contains empty string")

    @property
    def all_target_names(self) -> tuple[str, ...]:
        return self.closed_set_classes + self.open_vocab_descriptions


# --- predicates -------------------------------------------------------------
#
# Coordinates are normalized to [0, 1] x [0, 1] so predicates are portable
# across input resolutions. Pixel-coordinate variants can be added later as
# additional discriminated-union members if needed.


NormalizedPoint = tuple[float, float]


def _check_normalized_point(pt: NormalizedPoint, name: str) -> None:
    x, y = pt
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"{name}: point {pt!r} outside [0, 1]^2")


class ROIPredicate(BaseModel):
    """A target is inside a region of interest."""

    model_config = _frozen

    kind: Literal["roi"] = "roi"
    name: str = Field(min_length=1)
    polygon_normalized: tuple[NormalizedPoint, ...]
    target_classes: tuple[str, ...] = Field(min_length=1)

    def model_post_init(self, _context) -> None:
        if len(self.polygon_normalized) < 3:
            raise ValueError(f"ROIPredicate {self.name!r}: polygon needs at least 3 points")
        for pt in self.polygon_normalized:
            _check_normalized_point(pt, f"ROIPredicate {self.name!r}")


class LineCrossingDirection(str, Enum):
    EITHER = "either"
    START_TO_END = "start_to_end"
    END_TO_START = "end_to_start"


class LineCrossingPredicate(BaseModel):
    """A target's track crosses a line."""

    model_config = _frozen

    kind: Literal["line_crossing"] = "line_crossing"
    name: str = Field(min_length=1)
    line_start_normalized: NormalizedPoint
    line_end_normalized: NormalizedPoint
    direction: LineCrossingDirection
    target_classes: tuple[str, ...] = Field(min_length=1)

    def model_post_init(self, _context) -> None:
        _check_normalized_point(self.line_start_normalized, f"LineCrossingPredicate {self.name!r} start")
        _check_normalized_point(self.line_end_normalized, f"LineCrossingPredicate {self.name!r} end")
        if self.line_start_normalized == self.line_end_normalized:
            raise ValueError(f"LineCrossingPredicate {self.name!r}: degenerate line (start == end)")


class DwellTimePredicate(BaseModel):
    """A target stays inside a region for at least `min_seconds`."""

    model_config = _frozen

    kind: Literal["dwell_time"] = "dwell_time"
    name: str = Field(min_length=1)
    polygon_normalized: tuple[NormalizedPoint, ...]
    target_classes: tuple[str, ...] = Field(min_length=1)
    min_seconds: float = Field(gt=0)

    def model_post_init(self, _context) -> None:
        if len(self.polygon_normalized) < 3:
            raise ValueError(f"DwellTimePredicate {self.name!r}: polygon needs at least 3 points")
        for pt in self.polygon_normalized:
            _check_normalized_point(pt, f"DwellTimePredicate {self.name!r}")


Predicate = Annotated[
    Union[ROIPredicate, LineCrossingPredicate, DwellTimePredicate],
    Field(discriminator="kind"),
]


# --- operating point --------------------------------------------------------


class OperatingBias(str, Enum):
    """§1 — recall-weighted vs precision-weighted; balanced when neither is
    preferred."""

    RECALL_WEIGHTED = "recall_weighted"
    PRECISION_WEIGHTED = "precision_weighted"
    BALANCED = "balanced"


class OperatingPoint(BaseModel):
    """Bias tells the scorer which way to lean; explicit targets, when set,
    give tier-0 and tier-3 numbers to check against."""

    model_config = _frozen

    bias: OperatingBias
    target_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    target_precision: float | None = Field(default=None, ge=0.0, le=1.0)


# --- deployment hardware ----------------------------------------------------


class DeploymentHardware(BaseModel):
    """The hardware the pipeline will run on (§3, §13.1). The M4 calibration
    pass measures TargetCostMeasurement against this envelope; the tier-0 gate
    (§6) checks the measured latency against `latency_budget_ms` here.

    Note: shares HardwareEnvelope with ComponentManifest.ReferenceCost but the
    semantics differ. Manifest hardware = SKU used for reference measurements;
    contract hardware = deployment target."""

    model_config = _frozen

    hardware: HardwareEnvelope
    latency_budget_ms: float = Field(gt=0)
    min_throughput_fps: float | None = Field(default=None, gt=0)


# --- success criteria -------------------------------------------------------


class ScoringMetric(str, Enum):
    """Metrics tier-3 can compute against a GT eval set. Detection: recall,
    precision, F1, false-alarm rate. Tracking: MOTA, HOTA, IDF1."""

    RECALL = "recall"
    PRECISION = "precision"
    F1 = "f1"
    FALSE_ALARM_RATE = "false_alarm_rate"
    MOTA = "mota"
    HOTA = "hota"
    IDF1 = "idf1"


class SuccessCriterion(BaseModel):
    """A single pass/fail check applied to tier-3 output. `per_class` scopes
    the check to one target; None scopes it to the overall estimate."""

    model_config = _frozen

    metric: ScoringMetric
    per_class: str | None = None
    min_value: float | None = Field(default=None, ge=0.0, le=1.0)
    max_value: float | None = Field(default=None, ge=0.0, le=1.0)

    def model_post_init(self, _context) -> None:
        if self.min_value is None and self.max_value is None:
            raise ValueError(
                f"SuccessCriterion({self.metric}): must set at least one of min_value / max_value"
            )
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(
                f"SuccessCriterion({self.metric}): min_value {self.min_value} > max_value {self.max_value}"
            )


# --- the contract -----------------------------------------------------------


class TaskContract(BaseModel):
    """§3 spec-compiler output. Human-confirmed. Frozen once created."""

    model_config = _frozen

    name: str = Field(min_length=1)
    ontology: Ontology
    predicates: tuple[Predicate, ...] = Field(min_length=1)
    operating_point: OperatingPoint
    deployment_hardware: DeploymentHardware
    success_criteria: tuple[SuccessCriterion, ...] = Field(min_length=1)
    catalog_version: str = Field(min_length=1)

    def model_post_init(self, _context) -> None:
        known_targets = set(self.ontology.all_target_names)

        # Every predicate's target_classes must reference declared ontology entries.
        for pred in self.predicates:
            unknown = set(pred.target_classes) - known_targets
            if unknown:
                raise ValueError(
                    f"Predicate {pred.name!r}: target_classes {sorted(unknown)} "
                    f"not declared in ontology"
                )

        # Per-class success criteria must reference declared ontology entries.
        for crit in self.success_criteria:
            if crit.per_class is not None and crit.per_class not in known_targets:
                raise ValueError(
                    f"SuccessCriterion({crit.metric}): per_class {crit.per_class!r} "
                    f"not declared in ontology"
                )

        # Predicate names must be unique — the packager and event logic key on them.
        names = [p.name for p in self.predicates]
        if len(set(names)) != len(names):
            raise ValueError("Predicate names must be unique within a TaskContract")
