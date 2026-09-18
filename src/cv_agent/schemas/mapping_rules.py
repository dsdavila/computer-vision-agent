"""MappingRules — §4.2 compositional mapping, §16 #5 frozen interface.

Per-property rules composed by intersection, with an evidence-backed override
table for known joint interactions. Adding a topology or scene class is a
one-rule change against this schema, not a grid rewrite (§16 #5).

The schema is deliberately loose about component names (`"general_closed_set_v1"`
etc.) and scene-property field names (`"pixels_on_target"`, `"p50_px"`). Both
are cross-schema references validated at runtime by the planner — coupling
here to `ComponentManifest` or `SceneProperties` field names is brittle, and
migrations across catalog versions (§13) go smoother without it.

V0 ships with ~5 PropertyRules (one per constraining property) and an empty
override table. Overrides accumulate post-v0 as ledger evidence grows, and
their `ledger_refs` list pins that evidence.
"""

from pydantic import BaseModel, ConfigDict, Field

_frozen = ConfigDict(frozen=True, extra="forbid")


# --- catalog constraints ----------------------------------------------------


class CatalogConstraint(BaseModel):
    """Allowed component names per catalog axis. None on an axis means that
    axis is unconstrained by this rule; an empty tuple means nothing is
    allowed (which is either a hard refusal or a rule-author bug — the
    planner treats it as the former)."""

    model_config = _frozen

    detectors: tuple[str, ...] | None = None
    tiles: tuple[str, ...] | None = None
    trackers: tuple[str, ...] | None = None
    reid: tuple[str, ...] | None = None

    def constrains_any_axis(self) -> bool:
        return any(
            axis is not None
            for axis in (self.detectors, self.tiles, self.trackers, self.reid)
        )


# --- per-property rules -----------------------------------------------------


class ThresholdBand(BaseModel):
    """A half-open numeric band [min_value, max_value) on one scene-property
    metric, with the catalog constraint it implies. `None` on min means -inf;
    `None` on max means +inf, so a rule's bands can cover the whole real line
    with three or four entries."""

    model_config = _frozen

    name: str = Field(min_length=1)
    min_value: float | None = None
    max_value: float | None = None
    constraint: CatalogConstraint

    def model_post_init(self, _context) -> None:
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value >= self.max_value
        ):
            raise ValueError(
                f"ThresholdBand {self.name!r}: min_value {self.min_value} >= max_value {self.max_value}"
            )


class PropertyRule(BaseModel):
    """§4.2 — one rule per (scene property, metric). Bins the metric into
    non-overlapping bands, each carrying the catalog constraint that band
    implies. Composed with other property rules by intersection."""

    model_config = _frozen

    property: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    bands: tuple[ThresholdBand, ...] = Field(min_length=1)

    def model_post_init(self, _context) -> None:
        names = [b.name for b in self.bands]
        if len(set(names)) != len(names):
            raise ValueError(
                f"PropertyRule {self.property}.{self.metric}: band names not unique"
            )

        # Non-overlap check. Sort by min_value (None -> -inf) and verify each
        # band's max_value <= next band's min_value.
        neg_inf = float("-inf")
        pos_inf = float("inf")
        sorted_bands = sorted(
            self.bands,
            key=lambda b: neg_inf if b.min_value is None else b.min_value,
        )
        for prev, curr in zip(sorted_bands, sorted_bands[1:]):
            prev_max = pos_inf if prev.max_value is None else prev.max_value
            curr_min = neg_inf if curr.min_value is None else curr.min_value
            if prev_max > curr_min:
                raise ValueError(
                    f"PropertyRule {self.property}.{self.metric}: "
                    f"bands {prev.name!r} and {curr.name!r} overlap"
                )


# --- overrides --------------------------------------------------------------


class BandRef(BaseModel):
    """Reference to a specific band inside a PropertyRule. Used by Override
    conditions to name a joint scenario like 'pixels_on_target.p50_px is in
    the small_targets band AND congestion.occlusion_rate is in the heavy band'."""

    model_config = _frozen

    property: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    band_name: str = Field(min_length=1)


class Override(BaseModel):
    """§4.2 override entry. Joint interaction across multiple property bands,
    promoted only per the two-delta criterion — two single-variable deltas
    along the properties in question each failed (refuted or inconclusive) to
    explain the observed improvement. `ledger_refs` pin that evidence; the
    schema requires at least one, so an override cannot be added without it.

    Overrides added on correlation become permanent noise in a hand-authored
    rule set. That is a runtime discipline; the schema catches only the
    grossest violation (missing evidence)."""

    model_config = _frozen

    name: str = Field(min_length=1)
    # An override by definition spans multiple properties; a single-property
    # "override" is just a rule modification.
    conditions: tuple[BandRef, ...] = Field(min_length=2)
    constraint: CatalogConstraint
    ledger_refs: tuple[str, ...] = Field(min_length=1)

    def model_post_init(self, _context) -> None:
        if not self.constraint.constrains_any_axis():
            raise ValueError(
                f"Override {self.name!r}: constraint must constrain at least one axis "
                "(an override that does not restrict anything has no effect)"
            )


# --- the ruleset ------------------------------------------------------------


class MappingRules(BaseModel):
    """§4.2 compositional mapping — per-property rules + override table.
    Pinned to a catalog version (§13); migrations move rulesets across
    versions.

    Planner composition (M3 concern, not schema): apply each rule that fires
    for the measured scene, intersect the resulting constraints, then apply
    any override whose conditions are all satisfied. The result is the
    candidate topology set."""

    model_config = _frozen

    catalog_version: str = Field(min_length=1)
    rules: tuple[PropertyRule, ...] = Field(min_length=1)
    overrides: tuple[Override, ...] = ()

    def model_post_init(self, _context) -> None:
        keys = [(r.property, r.metric) for r in self.rules]
        if len(set(keys)) != len(keys):
            raise ValueError("MappingRules: multiple rules for the same (property, metric)")

        band_index: set[tuple[str, str, str]] = set()
        for rule in self.rules:
            for band in rule.bands:
                band_index.add((rule.property, rule.metric, band.name))

        for override in self.overrides:
            for cond in override.conditions:
                if (cond.property, cond.metric, cond.band_name) not in band_index:
                    raise ValueError(
                        f"Override {override.name!r}: condition "
                        f"{cond.property}.{cond.metric}={cond.band_name!r} "
                        "does not reference a declared band"
                    )
