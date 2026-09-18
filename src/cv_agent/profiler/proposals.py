"""Proposal-detector interface + runtime types.

The four proposal-dependent §4 probes (pixels-on-target, congestion,
appearance separability, target novelty) consume proposals from a detector.
This module defines the shape of those proposals and the detector protocol.

Runtime types, not schemas — they carry numpy arrays and per-frame values
that are computed on the fly. Nothing here is frozen for cross-version
storage; that job belongs to the M1 schemas.

Real detectors (open-vocab + SAM2 in M4.5; the existing VLM tuner in M3)
plug in by matching the ProposalDetector Protocol. No inheritance required.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from cv_agent.profiler.video import Frame


# --- boxes ------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in pixel coordinates. xyxy convention with
    x2 > x1 and y2 > y1 enforced at construction.

    Coordinates are floats, not ints — detectors typically emit float boxes.
    A probe that needs integer pixels rounds explicitly."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError(
                f"Box: expected x2 > x1 and y2 > y1, got "
                f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def iou(self, other: Box) -> float:
        """Intersection-over-union with another box. 0.0 for disjoint boxes."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


# --- proposals --------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """One detection proposal from a detector run against one frame.

    - `box`: pixel-coordinate xyxy in the frame's coordinate system.
    - `label`: the ontology target this proposal is claimed to be. For
      closed-set detectors, a class name from the detector's vocabulary; for
      open-vocab detectors, a natural-language target from the TaskContract
      ontology. String either way.
    - `score`: confidence in [0, 1] that `label` correctly identifies what is
      in `box`.
    - `embedding`: optional appearance embedding (numpy vector, dtype
      float32 by convention). None for detectors that do not produce one;
      the appearance-separability probe degrades to LOW confidence when
      embeddings are absent."""

    box: Box
    label: str
    score: float
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Proposal.score must be in [0, 1], got {self.score}")
        if not self.label:
            raise ValueError("Proposal.label must be non-empty")


@dataclass(frozen=True)
class FrameProposals:
    """All proposals for a single frame. An empty tuple means the detector
    found nothing (a valid, common outcome — probes treat it as data, not an
    error)."""

    frame: Frame
    proposals: tuple[Proposal, ...] = field(default_factory=tuple)


# --- detector interface -----------------------------------------------------


class ProposalDetector(Protocol):
    """Structural interface for anything that produces proposals.

    Batched by default — the primary method takes a list of frames because
    real detectors are much faster in batches. Callers that want single-frame
    behaviour call it with a length-1 list.

    `targets` names the ontology entries to look for. Closed-set detectors
    intersect this list with their trained vocabulary; targets that are
    unsupported are skipped (no proposals for that target), not errored —
    the target-novelty probe reads the resulting score distribution to
    infer coverage.
    """

    def propose(
        self,
        frames: Sequence[Frame],
        targets: Sequence[str],
    ) -> Sequence[FrameProposals]:
        """Run the detector over a batch. Returns one `FrameProposals` per
        input frame, in the same order."""
        ...
