"""EvalSet + DiagnosticSet — §7.1 split-set design, §16 #4 frozen interface.

Only EvalSet is one of the six frozen interfaces per §16. DiagnosticSet is
included here because §7.1 defines both together and M4.5 will produce this
shape; its schema may evolve when M4.5 lands, but the shape below is enough
for M3 to reason about.

**EvalSet is scored** — sampled on scene strata (§7.1), consumed by tier 3,
drives cross-topology selection and regression gating. Tier-3 API returns
per-stratum CIs (§7.2); the strata declared here are what those CIs key on.

**DiagnosticSet is never used for a scored number** — sampled on candidate-
config disagreement, used for tier-2 pruning, human spot-checks, and
failure-mode diagnosis. Keeping it out of any scored number is what lets it
be sampled aggressively.
"""

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

_frozen = ConfigDict(frozen=True, extra="forbid")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GroundTruthFormat(str, Enum):
    """v0 supports COCO for detection and MOT for tracking (§7, §17 M3)."""

    COCO = "coco"
    MOT = "mot"


class GroundTruthFile(BaseModel):
    """Reference to a ground-truth file on disk or in object storage. The
    schema does not load or parse the file — that is M3's job. Optional
    sha256 pins content; the M4 packager may require it at freeze time even
    though v0 lets the pilot produce GT however they like."""

    model_config = _frozen

    path: str = Field(min_length=1)
    format: GroundTruthFormat
    sha256: str | None = None

    def model_post_init(self, _context) -> None:
        if self.sha256 is not None and not _SHA256_RE.match(self.sha256):
            raise ValueError(
                f"GroundTruthFile.sha256 must be 64 lowercase hex characters, got {self.sha256!r}"
            )


class StratumLabel(BaseModel):
    """One scene stratum. Names are the keys tier-3 reports per-stratum CIs
    against. `description` is free-form for human review."""

    model_config = _frozen

    name: str = Field(min_length=1)
    description: str | None = None


class EvalSet(BaseModel):
    """§7.1 scored eval set. Sampled on scene strata (~6 in v0, §7.2).

    `item_strata` maps a GT-file item id (a COCO image_id or MOT sequence id)
    to a stratum declared in `strata`. Tier-3 groups scoring by these
    assignments; a stratum with no items is a shipping bug (§7.2 says empty
    strata are reported as no-estimate rather than dropped silently)."""

    model_config = _frozen

    name: str = Field(min_length=1)
    ground_truth: GroundTruthFile
    strata: tuple[StratumLabel, ...] = Field(min_length=1)
    item_strata: dict[str, str]
    scorer_id: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    task_contract_name: str = Field(min_length=1)

    def model_post_init(self, _context) -> None:
        names = [s.name for s in self.strata]
        if len(set(names)) != len(names):
            raise ValueError("EvalSet.strata contains duplicate names")

        known = set(names)
        unknown = {value for value in self.item_strata.values() if value not in known}
        if unknown:
            raise ValueError(
                f"EvalSet.item_strata references undeclared strata: {sorted(unknown)}"
            )

        if not self.item_strata:
            raise ValueError("EvalSet.item_strata must not be empty")


class DiagnosticSet(BaseModel):
    """§7.1 unscored diagnostic set. Sampled on candidate-config disagreement,
    used for tier-2 pruning and human failure-mode diagnosis. **Never used for
    a scored number** — that invariant is a design commitment (§7.1) and lives
    in whatever consumes this shape, not in the schema itself.

    `compared_configs` may be empty in v0 pre-M3 states (no candidates yet);
    once M3 populates it, the ids reference planner-emitted configs by their
    ledger config_id."""

    model_config = _frozen

    name: str = Field(min_length=1)
    ground_truth: GroundTruthFile
    catalog_version: str = Field(min_length=1)
    task_contract_name: str = Field(min_length=1)
    compared_configs: tuple[str, ...] = ()
