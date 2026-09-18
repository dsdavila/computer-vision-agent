"""Feasibility thresholds — §8, §13.1 catalog-derived and version-pinned.

The design's central discipline for the gate: threshold values must be
measured against the catalog and pinned to the catalog version that produced
them (§8). A hardcoded threshold goes silently wrong the first time a
component is added or retired (§13.1).

**v0 status.** The values below are declared, not yet measured. Real
derivation — running each detector at synthetic object scales, finding the
smallest size where each achieves usable recall, taking the minimum across
detectors — is threshold-derivation work that lands with the M2 harness
(task 8). Until then, these numbers are educated defaults tied to catalog
"v0-2026-09-17"; treat them as placeholders that will be replaced by
measured values.

`default_thresholds` raises on an unknown catalog version — that is the
enforcement point that keeps someone from silently proceeding on a catalog
we have not measured thresholds against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeasibilityThresholds:
    """Values the feasibility gate compares against. All fields are pinned to
    `catalog_version`; migrating catalogs means re-measuring the values and
    re-pinning."""

    catalog_version: str
    # Pixels-on-target floor. §8's canonical example ("~20 px") lives here.
    min_pixels_on_target_p50_px: float
    # Congestion ceiling. Two independent bounds — hitting either refuses.
    max_congestion_detections_per_frame: float
    max_congestion_occlusion_rate: float


# v0 threshold table. Placeholder values pending the derivation harness.
_V0_2026_09_17 = FeasibilityThresholds(
    catalog_version="v0-2026-09-17",
    min_pixels_on_target_p50_px=20.0,
    max_congestion_detections_per_frame=40.0,
    max_congestion_occlusion_rate=0.6,
)


_TABLE: dict[str, FeasibilityThresholds] = {
    "v0-2026-09-17": _V0_2026_09_17,
}


def default_thresholds(catalog_version: str) -> FeasibilityThresholds:
    """Return the pinned threshold table for a catalog version.

    Raises if no thresholds have been declared for the given version — that
    is the forcing function that keeps the system from silently proceeding
    on an unmeasured catalog."""
    try:
        return _TABLE[catalog_version]
    except KeyError:
        known = ", ".join(sorted(_TABLE)) or "(none)"
        raise ValueError(
            f"No feasibility thresholds pinned for catalog {catalog_version!r}. "
            f"Known: {known}. Add an entry to feasibility/thresholds.py."
        ) from None
