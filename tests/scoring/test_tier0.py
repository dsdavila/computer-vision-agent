"""Tier-0 filter tests — §6 hard constraints, §13.1 target-cost discipline."""

from __future__ import annotations

from cv_agent.schemas import Ontology
from cv_agent.scoring import (
    DEFAULT_LICENSE_POLICY,
    LicensePolicy,
    tier0_filter,
    tier0_survivors,
)

from .conftest import (
    make_detector,
    make_target_costs,
    make_task_contract,
    make_topology,
)


# --- happy path -------------------------------------------------------------


def test_healthy_topology_passes():
    t = make_topology()
    result = tier0_filter(t, make_task_contract())
    assert result.passed
    assert result.failures == ()


# --- latency ---------------------------------------------------------------


def test_latency_budget_exceeded():
    t = make_topology()
    costs = make_target_costs(latency_by_component={"general_v1": 80.0, "motion_only_v1": 80.0})
    result = tier0_filter(t, make_task_contract(latency_budget_ms=100.0), target_costs=costs)
    assert result.refused
    assert {f.constraint for f in result.failures} == {"latency_budget"}
    msg = result.failures[0].message
    assert "exceeds budget" in msg
    assert "general_v1=80.0ms" in msg


def test_latency_within_budget_passes():
    t = make_topology()
    costs = make_target_costs()  # defaults 15ms per component = 60ms total
    result = tier0_filter(t, make_task_contract(latency_budget_ms=100.0), target_costs=costs)
    assert result.passed


def test_latency_check_skipped_when_target_costs_absent():
    """§13.1: tier-0 gates on target-measured cost only. Without measurements,
    the check does not fire — even if the topology would exceed budget."""
    t = make_topology()
    result = tier0_filter(t, make_task_contract(latency_budget_ms=0.001), target_costs=None)
    # Budget is 1μs; without measurements, no latency failure.
    assert "latency_budget" not in {f.constraint for f in result.failures}


def test_latency_check_skipped_when_any_component_measurement_missing():
    t = make_topology()
    full = make_target_costs()
    del full["reid_off_v1"]  # partial measurements
    result = tier0_filter(
        t,
        make_task_contract(latency_budget_ms=0.001),
        target_costs=full,
    )
    assert "latency_budget" not in {f.constraint for f in result.failures}


# --- VRAM -----------------------------------------------------------------


def test_vram_budget_exceeded():
    t = make_topology()
    # Sum defaults to 4 * 1200MB = 4800MB. Budget of 2GB (2048MB) → exceeded.
    result = tier0_filter(t, make_task_contract(vram_gb=2), target_costs=make_target_costs())
    assert "vram_budget" in {f.constraint for f in result.failures}


def test_vram_within_budget():
    t = make_topology()
    # Sum defaults ≈ 4800MB. Budget of 24GB (24576MB) → fine.
    result = tier0_filter(t, make_task_contract(vram_gb=24), target_costs=make_target_costs())
    assert "vram_budget" not in {f.constraint for f in result.failures}


# --- License -------------------------------------------------------------


def test_disallowed_license_fails():
    t = make_topology(detector=make_detector(license="AGPL-3.0"))
    result = tier0_filter(t, make_task_contract())
    failures = {(f.constraint, f.message) for f in result.failures}
    assert any(c == "license" for c, _ in failures)
    assert any("AGPL-3.0" in m for _, m in failures)


def test_strict_policy_rejects_apache():
    """A caller-supplied license policy overrides the default."""
    strict = LicensePolicy(allowed=frozenset({"MIT"}))
    result = tier0_filter(make_topology(), make_task_contract(), license_policy=strict)
    # Every component is Apache-2.0; strict policy rejects all four.
    licenses_flagged = [f for f in result.failures if f.constraint == "license"]
    assert len(licenses_flagged) == 4


def test_default_policy_allows_apache_and_mit_and_bsd():
    assert "Apache-2.0" in DEFAULT_LICENSE_POLICY.allowed
    assert "MIT" in DEFAULT_LICENSE_POLICY.allowed
    assert "BSD-3-Clause" in DEFAULT_LICENSE_POLICY.allowed


# --- Vocabulary coverage --------------------------------------------------


def test_open_vocab_detector_covers_everything():
    open_vocab = make_detector(supported_classes=None)
    t = make_topology(detector=open_vocab)
    ontology = Ontology(
        closed_set_classes=("person",),
        open_vocab_descriptions=("delivery truck",),
    )
    result = tier0_filter(t, make_task_contract(ontology=ontology))
    assert "vocabulary_coverage" not in {f.constraint for f in result.failures}


def test_closed_set_detector_missing_class_fails():
    t = make_topology(detector=make_detector(supported_classes=("person",)))
    ontology = Ontology(closed_set_classes=("person", "vehicle"))
    result = tier0_filter(t, make_task_contract(ontology=ontology))
    vocab = [f for f in result.failures if f.constraint == "vocabulary_coverage"]
    assert len(vocab) == 1
    assert "vehicle" in vocab[0].message


def test_closed_set_detector_with_open_vocab_ontology_fails():
    t = make_topology(detector=make_detector(supported_classes=("person",)))
    ontology = Ontology(
        closed_set_classes=("person",),
        open_vocab_descriptions=("delivery truck",),
    )
    result = tier0_filter(t, make_task_contract(ontology=ontology))
    vocab = [f for f in result.failures if f.constraint == "vocabulary_coverage"]
    assert len(vocab) == 1
    assert "delivery truck" in vocab[0].message


# --- Multiple failures -----------------------------------------------------


def test_multiple_failures_reported():
    """Bad-fit topology fails on several constraints; all fire, none short-circuit."""
    t = make_topology(
        detector=make_detector(
            license="AGPL-3.0",
            supported_classes=("dog",),  # doesn't cover the ontology
        )
    )
    result = tier0_filter(t, make_task_contract())
    kinds = {f.constraint for f in result.failures}
    assert {"license", "vocabulary_coverage"}.issubset(kinds)


# --- Survivors convenience -------------------------------------------------


def test_survivors_returns_only_passing():
    good = make_topology()
    bad = make_topology(detector=make_detector(license="AGPL-3.0"))
    contract = make_task_contract()
    survivors = tier0_survivors([good, bad], contract)
    assert survivors == [good]
