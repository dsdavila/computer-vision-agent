"""Tier-0 filter — §6 hard constraints, filtering only.

§6 places tier 0 as a filter, not a scorer. A candidate topology either
passes or fails; there is no score. Failing candidates are dropped before
tier 1/3 run.

Constraints:

- **Latency budget** (§13.1). Sum of target-measured latencies across the
  four components must fit inside `TaskContract.deployment_hardware.latency_budget_ms`.
  Only fires when target-cost measurements are provided; §13.1 forbids
  gating on reference-scaled cost. Missing measurements → check is skipped
  silently. Callers who need cost-gated results provide the calibration
  outputs.
- **VRAM budget** (§13.1). Same pattern as latency. Uses SUM across
  components as a conservative bound — assumes all four models are
  resident simultaneously. Real deployments load/unload; a peak-tracking
  bound can replace this once we track model residency.
- **License policy**. Each component's license must appear in the caller's
  allowed set. Default policy is permissive (Apache-2.0 / MIT / BSD).
  Restrictive deployments pass a stricter policy.
- **Vocabulary coverage**. The detector must either be open-vocab
  (supported_classes=None) or its closed-set vocabulary must cover every
  target in the TaskContract ontology. Open-vocab descriptions on the
  ontology require an open-vocab detector.

Airgap is deferred — ComponentManifest.preconditions does not yet carry a
network-requirement field. §16 #3 flags cost as the load-bearing manifest
change; airgap can layer on later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cv_agent.schemas import (
    ComponentManifest,
    Ontology,
    TargetCostMeasurement,
    TaskContract,
)
from cv_agent.scoring.topology import Topology


# --- license policy --------------------------------------------------------


@dataclass(frozen=True)
class LicensePolicy:
    """Allowed component licenses. Empty set means no component licenses
    are accepted (used for testing hard failures)."""

    allowed: frozenset[str]


DEFAULT_LICENSE_POLICY = LicensePolicy(
    allowed=frozenset(
        {
            "Apache-2.0",
            "MIT",
            "BSD-2-Clause",
            "BSD-3-Clause",
        }
    ),
)


# --- result types ----------------------------------------------------------


@dataclass(frozen=True)
class Tier0Failure:
    constraint: str
    message: str


@dataclass(frozen=True)
class Tier0Result:
    topology: Topology
    passed: bool
    failures: tuple[Tier0Failure, ...]

    @property
    def refused(self) -> bool:
        return not self.passed


# --- public entrypoints ----------------------------------------------------


def tier0_filter(
    topology: Topology,
    task_contract: TaskContract,
    *,
    target_costs: dict[str, TargetCostMeasurement] | None = None,
    license_policy: LicensePolicy | None = None,
) -> Tier0Result:
    """Run all applicable §6 tier-0 constraints against one Topology.

    Latency and VRAM checks fire only if `target_costs` covers every
    component in the topology. §13.1 forbids gating on reference-scaled
    cost; missing measurements → skip silently rather than fall back."""

    policy = license_policy or DEFAULT_LICENSE_POLICY
    failures: list[Tier0Failure] = []

    if target_costs is not None:
        latency_or_missing = _resolve_target_costs(topology, target_costs)
        if latency_or_missing is not None:
            failures.extend(
                _check_latency(topology, task_contract, latency_or_missing)
            )
            failures.extend(
                _check_vram(topology, task_contract, latency_or_missing)
            )

    failures.extend(_check_license(topology, policy))
    failures.extend(_check_vocabulary(topology, task_contract.ontology))

    return Tier0Result(
        topology=topology,
        passed=not failures,
        failures=tuple(failures),
    )


def tier0_survivors(
    topologies: Sequence[Topology],
    task_contract: TaskContract,
    *,
    target_costs: dict[str, TargetCostMeasurement] | None = None,
    license_policy: LicensePolicy | None = None,
) -> list[Topology]:
    """Convenience: return only the topologies that pass tier-0."""
    return [
        t
        for t in topologies
        if tier0_filter(
            t,
            task_contract,
            target_costs=target_costs,
            license_policy=license_policy,
        ).passed
    ]


# --- individual checks -----------------------------------------------------


def _resolve_target_costs(
    topology: Topology,
    target_costs: dict[str, TargetCostMeasurement],
) -> dict[str, TargetCostMeasurement] | None:
    """Return the subset of target_costs relevant to this topology, or
    None if any component's measurement is missing (skip the cost checks
    silently — §13.1 forbids partial gating)."""
    resolved: dict[str, TargetCostMeasurement] = {}
    for component in topology.components():
        cost = target_costs.get(component.name)
        if cost is None:
            return None
        resolved[component.name] = cost
    return resolved


def _check_latency(
    topology: Topology,
    contract: TaskContract,
    costs: dict[str, TargetCostMeasurement],
) -> list[Tier0Failure]:
    total_ms = sum(costs[c.name].latency_ms for c in topology.components())
    budget = contract.deployment_hardware.latency_budget_ms
    if total_ms > budget:
        breakdown = ", ".join(
            f"{c.name}={costs[c.name].latency_ms:.1f}ms"
            for c in topology.components()
        )
        return [
            Tier0Failure(
                constraint="latency_budget",
                message=(
                    f"Topology total latency {total_ms:.1f}ms exceeds budget "
                    f"{budget:.1f}ms. Breakdown: {breakdown}. §13.1: measured on "
                    "target hardware."
                ),
            )
        ]
    return []


def _check_vram(
    topology: Topology,
    contract: TaskContract,
    costs: dict[str, TargetCostMeasurement],
) -> list[Tier0Failure]:
    total_mb = sum(costs[c.name].vram_mb for c in topology.components())
    budget_mb = contract.deployment_hardware.hardware.vram_gb * 1024
    if total_mb > budget_mb:
        breakdown = ", ".join(
            f"{c.name}={costs[c.name].vram_mb:.0f}MB"
            for c in topology.components()
        )
        return [
            Tier0Failure(
                constraint="vram_budget",
                message=(
                    f"Topology total VRAM {total_mb:.0f}MB exceeds budget "
                    f"{budget_mb:.0f}MB. Breakdown: {breakdown}. Conservative "
                    "sum bound; peak-only estimate can replace this once "
                    "residency is tracked."
                ),
            )
        ]
    return []


def _check_license(
    topology: Topology, policy: LicensePolicy
) -> list[Tier0Failure]:
    failures: list[Tier0Failure] = []
    for component in topology.components():
        if component.license not in policy.allowed:
            failures.append(
                Tier0Failure(
                    constraint="license",
                    message=(
                        f"Component {component.name!r} has license "
                        f"{component.license!r} not in allowed set "
                        f"{sorted(policy.allowed)}."
                    ),
                )
            )
    return failures


def _check_vocabulary(
    topology: Topology, ontology: Ontology
) -> list[Tier0Failure]:
    detector: ComponentManifest = topology.detector
    supported = detector.preconditions.supported_classes
    if supported is None:
        # Open-vocab detector covers everything.
        return []

    supported_set = set(supported)
    missing_closed = [
        name
        for name in ontology.closed_set_classes
        if name not in supported_set
    ]
    if missing_closed:
        return [
            Tier0Failure(
                constraint="vocabulary_coverage",
                message=(
                    f"Detector {detector.name!r} is closed-set and does not "
                    f"cover ontology classes: {sorted(missing_closed)}. "
                    f"Supports: {sorted(supported_set)}."
                ),
            )
        ]

    if ontology.open_vocab_descriptions:
        return [
            Tier0Failure(
                constraint="vocabulary_coverage",
                message=(
                    f"Detector {detector.name!r} is closed-set but the "
                    f"ontology includes open-vocab targets: "
                    f"{list(ontology.open_vocab_descriptions)}. Pick an "
                    "open-vocab detector for this task."
                ),
            )
        ]
    return []
