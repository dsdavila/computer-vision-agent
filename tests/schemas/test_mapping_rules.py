"""MappingRules schema tests — bands, non-overlap, override discipline."""

import pytest
from pydantic import ValidationError

from cv_agent.schemas import (
    BandRef,
    CatalogConstraint,
    MappingRules,
    Override,
    PropertyRule,
    ThresholdBand,
)


def _pixels_rule() -> PropertyRule:
    return PropertyRule(
        property="pixels_on_target",
        metric="p50_px",
        bands=(
            ThresholdBand(
                name="small_targets",
                min_value=None,
                max_value=20.0,
                constraint=CatalogConstraint(
                    detectors=("small_object_v1",),
                    tiles=("on",),
                ),
            ),
            ThresholdBand(
                name="medium_targets",
                min_value=20.0,
                max_value=60.0,
                constraint=CatalogConstraint(
                    detectors=("general_closed_set_v1", "small_object_v1"),
                    tiles=("off",),
                ),
            ),
            ThresholdBand(
                name="large_targets",
                min_value=60.0,
                max_value=None,
                constraint=CatalogConstraint(
                    detectors=("general_closed_set_v1", "open_vocab_v1"),
                    tiles=("off",),
                ),
            ),
        ),
    )


def _congestion_rule() -> PropertyRule:
    return PropertyRule(
        property="congestion",
        metric="occlusion_rate",
        bands=(
            ThresholdBand(
                name="low_occlusion",
                max_value=0.3,
                constraint=CatalogConstraint(
                    trackers=("motion_only_v1", "appearance_assisted_v1"),
                ),
            ),
            ThresholdBand(
                name="heavy_occlusion",
                min_value=0.3,
                constraint=CatalogConstraint(trackers=("appearance_assisted_v1",)),
            ),
        ),
    )


def _valid_rules() -> MappingRules:
    return MappingRules(
        catalog_version="v0-2026-09-17",
        rules=(_pixels_rule(), _congestion_rule()),
    )


# --- happy path -------------------------------------------------------------


def test_valid_ruleset_constructs():
    r = _valid_rules()
    assert len(r.rules) == 2
    assert r.overrides == ()


def test_empty_override_table_default():
    r = _valid_rules()
    assert r.overrides == ()


def test_valid_override_with_ledger_refs():
    r = MappingRules(
        catalog_version="v0-2026-09-17",
        rules=(_pixels_rule(), _congestion_rule()),
        overrides=(
            Override(
                name="small_and_heavy",
                conditions=(
                    BandRef(property="pixels_on_target", metric="p50_px", band_name="small_targets"),
                    BandRef(property="congestion", metric="occlusion_rate", band_name="heavy_occlusion"),
                ),
                constraint=CatalogConstraint(reid=("on",)),
                ledger_refs=("ledger/0042.json", "ledger/0057.json"),
            ),
        ),
    )
    assert r.overrides[0].name == "small_and_heavy"


# --- band invariants --------------------------------------------------------


def test_band_min_greater_than_max_rejected():
    with pytest.raises(ValueError):
        ThresholdBand(
            name="reversed",
            min_value=20.0,
            max_value=10.0,
            constraint=CatalogConstraint(),
        )


def test_band_equal_min_max_rejected():
    with pytest.raises(ValueError):
        ThresholdBand(
            name="empty",
            min_value=20.0,
            max_value=20.0,
            constraint=CatalogConstraint(),
        )


def test_bands_overlapping_rejected():
    with pytest.raises(ValueError):
        PropertyRule(
            property="p",
            metric="m",
            bands=(
                ThresholdBand(name="a", min_value=0.0, max_value=15.0, constraint=CatalogConstraint()),
                ThresholdBand(name="b", min_value=10.0, max_value=20.0, constraint=CatalogConstraint()),
            ),
        )


def test_bands_touching_at_boundary_allowed():
    # Half-open [min, max) — a band ending at 20 and another starting at 20 don't overlap.
    rule = PropertyRule(
        property="p",
        metric="m",
        bands=(
            ThresholdBand(name="a", min_value=0.0, max_value=20.0, constraint=CatalogConstraint()),
            ThresholdBand(name="b", min_value=20.0, max_value=40.0, constraint=CatalogConstraint()),
        ),
    )
    assert len(rule.bands) == 2


def test_bands_with_open_ends_allowed():
    # Both ends open: covers the whole real line.
    rule = PropertyRule(
        property="p",
        metric="m",
        bands=(
            ThresholdBand(name="low", max_value=10.0, constraint=CatalogConstraint()),
            ThresholdBand(name="mid", min_value=10.0, max_value=90.0, constraint=CatalogConstraint()),
            ThresholdBand(name="high", min_value=90.0, constraint=CatalogConstraint()),
        ),
    )
    assert len(rule.bands) == 3


def test_band_names_must_be_unique_within_rule():
    with pytest.raises(ValueError):
        PropertyRule(
            property="p",
            metric="m",
            bands=(
                ThresholdBand(name="x", max_value=10.0, constraint=CatalogConstraint()),
                ThresholdBand(name="x", min_value=10.0, constraint=CatalogConstraint()),
            ),
        )


# --- ruleset invariants -----------------------------------------------------


def test_duplicate_property_metric_rejected():
    with pytest.raises(ValueError):
        MappingRules(
            catalog_version="v0",
            rules=(_pixels_rule(), _pixels_rule()),
        )


def test_override_condition_must_reference_declared_band():
    with pytest.raises(ValueError):
        MappingRules(
            catalog_version="v0",
            rules=(_pixels_rule(),),
            overrides=(
                Override(
                    name="bad",
                    conditions=(
                        BandRef(property="pixels_on_target", metric="p50_px", band_name="small_targets"),
                        BandRef(property="congestion", metric="occlusion_rate", band_name="heavy_occlusion"),
                    ),
                    constraint=CatalogConstraint(reid=("on",)),
                    ledger_refs=("ledger/1.json",),
                ),
            ),
        )


# --- override discipline ----------------------------------------------------


def test_override_requires_ledger_refs():
    with pytest.raises(ValidationError):
        Override(
            name="unproven",
            conditions=(
                BandRef(property="pixels_on_target", metric="p50_px", band_name="small_targets"),
                BandRef(property="congestion", metric="occlusion_rate", band_name="heavy_occlusion"),
            ),
            constraint=CatalogConstraint(reid=("on",)),
            ledger_refs=(),  # empty — the schema enforces §4.2 promotion criterion
        )


def test_override_requires_at_least_two_conditions():
    with pytest.raises(ValidationError):
        Override(
            name="single",
            conditions=(
                BandRef(property="pixels_on_target", metric="p50_px", band_name="small_targets"),
            ),
            constraint=CatalogConstraint(reid=("on",)),
            ledger_refs=("ledger/1.json",),
        )


def test_override_must_constrain_some_axis():
    with pytest.raises(ValueError):
        Override(
            name="noop",
            conditions=(
                BandRef(property="pixels_on_target", metric="p50_px", band_name="small_targets"),
                BandRef(property="congestion", metric="occlusion_rate", band_name="heavy_occlusion"),
            ),
            constraint=CatalogConstraint(),  # all axes None — no restriction
            ledger_refs=("ledger/1.json",),
        )


# --- schema-drift guards ----------------------------------------------------


def test_frozen():
    r = _valid_rules()
    with pytest.raises(ValidationError):
        r.catalog_version = "v1"  # type: ignore[misc]


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        MappingRules.model_validate(
            _valid_rules().model_dump() | {"undocumented": 1}
        )


def test_rules_required():
    with pytest.raises(ValidationError):
        MappingRules(catalog_version="v0", rules=())


def test_catalog_version_required():
    with pytest.raises(ValidationError):
        MappingRules(catalog_version="", rules=(_pixels_rule(),))
