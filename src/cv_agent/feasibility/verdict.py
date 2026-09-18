"""FeasibilityVerdict types — §8 output shape.

Distinct from schemas.Verdict (§9), which records the outcome of a
numeric hypothesis. This is a proceed/refuse verdict on whether the
pipeline should try to solve the problem at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeasibilityOutcome(str, Enum):
    PROCEED = "proceed"
    REFUSE = "refuse"


@dataclass(frozen=True)
class RefusalReason:
    """One refusal cited by the feasibility gate. `rule` is the rule id (so
    ledger entries and downstream code can filter); `message` is the
    human-readable narrative in the style of §8's canonical example."""

    rule: str
    message: str


@dataclass(frozen=True)
class FeasibilityVerdict:
    """§8 gate output. If `outcome` is PROCEED, `reasons` is empty. If
    REFUSE, `reasons` has one entry per firing rule."""

    outcome: FeasibilityOutcome
    reasons: tuple[RefusalReason, ...]
    catalog_version: str

    @property
    def passed(self) -> bool:
        return self.outcome is FeasibilityOutcome.PROCEED

    @property
    def refused(self) -> bool:
        return self.outcome is FeasibilityOutcome.REFUSE
