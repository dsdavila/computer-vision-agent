"""LedgerEntry — §9 decision ledger, §16 #2 frozen interface.

Append-only. The winning config is a pointer into the ledger, not the artifact
itself. Negative results survive — the branches that failed and why *are* the
search pruning (§9).

`hypothesis` + `verdict` is what makes the ledger worth keeping (§9). A later
run can ask "in this regime, did raising the spawn threshold historically
help?" and answer from evidence rather than from a prior. That is only true
if hypothesis is stated numerically BEFORE the run and verdict is COMPUTED
from hypothesis + outcome, not narrated. This schema keeps that discipline
enforceable: hypothesis is a Prediction with baseline + predicted delta +
tolerance, and verdict is an enum whose computation lives in M3, not here.

Two renderings, one store (§9): structured fields for the typed record;
`narrative` string for the human rendering. Both required — the narrative is
generated at decision time so it does not depend on whatever model is
available later.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from cv_agent.schemas.component_manifest import (
    ConfigValue,
    CostMeasurement,
    TargetCostMeasurement,
)
from cv_agent.schemas.scene_properties import SceneProperties

_frozen = ConfigDict(frozen=True, extra="forbid")


# --- enums ------------------------------------------------------------------


class ScorerTier(int, Enum):
    """§6 tiers. Recorded per entry so trust in the score is inspectable."""

    HARD_CONSTRAINT = 0
    LABEL_FREE_PROXY = 1
    VLM_PAIRWISE = 2
    GROUND_TRUTH = 3


class Verdict(str, Enum):
    """§9 — computed, not narrated. Confirmed / refuted / inconclusive against
    the numeric hypothesis + tolerance."""

    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


# --- payload shapes ---------------------------------------------------------


class MetricReading(BaseModel):
    """One named numeric measurement. Used for `evidence` and `outcome`.
    Tier-3 readings carry per-stratum CIs (§7.2); other tiers leave them None."""

    model_config = _frozen

    name: str = Field(min_length=1)
    value: float
    stratum: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    def model_post_init(self, _context) -> None:
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError(
                f"MetricReading {self.name!r}: ci_low {self.ci_low} > ci_high {self.ci_high}"
            )


class Prediction(BaseModel):
    """§9 — hypothesis stated numerically before the run. Baseline + delta +
    tolerance is what makes the verdict computable:
      outcome_delta = outcome_value - baseline_value
      confirmed if outcome_delta in [predicted_delta - tolerance, predicted_delta + tolerance]
      refuted if sign(outcome_delta) != sign(predicted_delta) and |outcome_delta| > tolerance
      inconclusive otherwise
    Computation lives in M3, not here — the schema captures the shape."""

    model_config = _frozen

    metric_name: str = Field(min_length=1)
    baseline_value: float
    predicted_delta: float
    tolerance: float = Field(gt=0)
    stratum: str | None = None


class Provenance(BaseModel):
    """§9 — model id, version, seed, catalog version. Everything a re-run
    would need to check reproducibility."""

    model_config = _frozen

    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    seed: int
    catalog_version: str = Field(min_length=1)


# --- the entry --------------------------------------------------------------


class LedgerEntry(BaseModel):
    """§9 decision record. Frozen once written; ledger is append-only (§9.1)."""

    model_config = _frozen

    id: str = Field(min_length=1)
    created_at: datetime
    parent_entry_id: str | None = None
    task_contract_name: str = Field(min_length=1)

    # --- config state ---
    # Flat dict of dotted keys, e.g. "topology.detector", "detector.confidence_threshold".
    full_config: dict[str, ConfigValue]
    # Parameter names that changed vs the parent entry (or "" if a root entry).
    # Explicit so the case base can filter cheaply and `confounded` is derivable.
    changed_parameters: tuple[str, ...]

    # --- scene at time of decision ---
    scene_properties: SceneProperties

    # --- reasoning ---
    hypothesis: tuple[Prediction, ...] = Field(min_length=1)
    evidence: tuple[MetricReading, ...] = Field(min_length=1)
    outcome: tuple[MetricReading, ...] = Field(min_length=1)
    verdict: Verdict

    # --- scoring & confounding ---
    scorer_tier: ScorerTier
    confounded: bool

    # --- cost (§13.1) ---
    cost_estimated: CostMeasurement
    cost_measured: TargetCostMeasurement | None = None

    # --- provenance & rendering ---
    provenance: Provenance
    narrative: str = Field(min_length=1)

    def model_post_init(self, _context) -> None:
        if self.parent_entry_id is not None and self.parent_entry_id == self.id:
            raise ValueError(f"LedgerEntry {self.id!r}: parent_entry_id must differ from id")

        # §9 — multi-variable changes are always confounded.
        if len(self.changed_parameters) >= 2 and not self.confounded:
            raise ValueError(
                f"LedgerEntry {self.id!r}: len(changed_parameters)={len(self.changed_parameters)} "
                "requires confounded=True (§9)"
            )

        # Every parameter said to have changed must exist in the snapshot.
        missing = [p for p in self.changed_parameters if p not in self.full_config]
        if missing:
            raise ValueError(
                f"LedgerEntry {self.id!r}: changed_parameters {missing} not present in full_config"
            )

        # changed_parameters uniqueness — a parameter can't change twice in one entry.
        if len(set(self.changed_parameters)) != len(self.changed_parameters):
            raise ValueError(
                f"LedgerEntry {self.id!r}: changed_parameters contains duplicates"
            )
