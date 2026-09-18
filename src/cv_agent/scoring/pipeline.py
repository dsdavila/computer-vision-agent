"""Runtime types for pipeline output — what tier-1 and tier-3 consume.

Not schemas. These live only during a scoring run; nothing here is stored
across catalog versions. The persisted record is the LedgerEntry.

Track and TrackedFrame are the tracking side (used by short_track_ratio and
track_count_stability). PipelineOutput bundles a full run: per-frame
detections + tracks + the detector threshold that produced them (needed for
conf_p50_margin and near_threshold_fraction).
"""

from __future__ import annotations

from dataclasses import dataclass

from cv_agent.profiler.proposals import Box, FrameProposals


@dataclass(frozen=True)
class TrackedFrame:
    """One tracked observation of an identity in one frame."""

    frame_index: int
    box: Box
    score: float


@dataclass(frozen=True)
class Track:
    """One identity's trajectory. `frames` is per-frame observations in
    frame_index order — no gaps required (a track can drop and reacquire).
    Lifespan is len(frames), which is what short_track_ratio keys on."""

    id: int
    frames: tuple[TrackedFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError(f"Track {self.id}: frames must be non-empty")

    @property
    def lifespan_frames(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class PipelineOutput:
    """One topology + config's run over one video. Consumed by tier-1
    proxies and by tier-3 scoring functions."""

    frame_proposals: tuple[FrameProposals, ...]
    tracks: tuple[Track, ...]
    detector_threshold: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.detector_threshold <= 1.0):
            raise ValueError(
                f"PipelineOutput.detector_threshold must be in [0, 1], "
                f"got {self.detector_threshold}"
            )
