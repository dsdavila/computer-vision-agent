"""Topology — runtime representation of a specific pipeline configuration.

Per §15, v0 has four catalog axes (detector, tile, tracker, reid). A
Topology carries the ComponentManifest for each axis plus a flat parameter
config (dotted keys, matching LedgerEntry.full_config). Runtime type, not a
schema — no cross-version storage is expected of Topology objects.

Kind validation is enforced at construction: pass the wrong axis's manifest
and you get a ValueError. This is cheap tape that prevents an easy category
error from silently propagating downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cv_agent.schemas import ComponentKind, ComponentManifest, ConfigValue


@dataclass(frozen=True)
class Topology:
    """A specific (detector, tile, tracker, reid) selection + parameter config."""

    detector: ComponentManifest
    tile: ComponentManifest
    tracker: ComponentManifest
    reid: ComponentManifest
    config: dict[str, ConfigValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected: dict[ComponentKind, str] = {
            ComponentKind.DETECTOR: "detector",
            ComponentKind.TILE: "tile",
            ComponentKind.TRACKER: "tracker",
            ComponentKind.REID: "reid",
        }
        for slot, wanted_name in expected.items():
            got: ComponentManifest = getattr(self, wanted_name)
            if got.kind is not slot:
                raise ValueError(
                    f"Topology.{wanted_name} expected kind={slot.value!r}, "
                    f"got manifest {got.name!r} with kind={got.kind.value!r}"
                )

    def components(self) -> tuple[ComponentManifest, ComponentManifest, ComponentManifest, ComponentManifest]:
        """The four component manifests in axis order (detector, tile, tracker, reid)."""
        return (self.detector, self.tile, self.tracker, self.reid)

    def component_names(self) -> tuple[str, str, str, str]:
        return tuple(m.name for m in self.components())  # type: ignore[return-value]

    def describe(self) -> str:
        """Short human-readable identifier used in refusal messages and logs."""
        return " × ".join(self.component_names())

    def __hash__(self) -> int:
        # Frozen dataclass would generate a __hash__, but `config` is a dict
        # (unhashable) — the generated one raises TypeError on hash(). Hash
        # on component names + a canonical tuple of config items so
        # Topologies can be used as set / dict keys.
        cfg = tuple(sorted(self.config.items()))
        return hash((self.component_names(), cfg))
