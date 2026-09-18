"""Lighting probe — §4 detector-independent property (§4.1).

Emits:
- intensity_variance_over_time: how much the frame-level mean intensity
  varies across sampled frames. High → lighting changes across the video
  (day/night, indoor/outdoor transitions); low → stable lighting.
- saturation_clipping_fraction: fraction of pixels at either end of the
  8-bit intensity range across all sampled frames. High → the camera is
  clipping and appearance features are unreliable.

Confidence follows sample size only. Lighting is detector-independent, so
confidence does not reflect proposal quality — it reflects whether the
sampler gave the probe enough frames to speak with confidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from cv_agent.profiler.video import Frame
from cv_agent.schemas import Lighting, ProbeConfidence


# Confidence tiers keyed on sample size. Values are opinionated but not
# catalog-dependent — the sampler is what decides frame counts, and the probe
# reports on whether that sample was enough to be meaningful.
_HIGH_MIN_FRAMES = 20
_MEDIUM_MIN_FRAMES = 8

# 8-bit clipping thresholds. Pixels at the extreme ends of the intensity
# range are treated as saturated (near-white) or crushed (near-black).
_LOW_CLIP = 5
_HIGH_CLIP = 250


def measure_lighting(frames: Sequence[Frame]) -> Lighting:
    """Detector-independent lighting probe.

    Runs in O(F * H * W) — one grayscale conversion + one comparison scan per
    frame. Frames are treated read-only; probe does not mutate `frame.array`."""
    if not frames:
        raise ValueError("measure_lighting: frames must be non-empty")

    per_frame_mean = np.empty(len(frames), dtype=np.float64)
    clipped_total = 0
    pixel_total = 0
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
        per_frame_mean[i] = float(gray.mean())
        clipped = int(((gray <= _LOW_CLIP) | (gray >= _HIGH_CLIP)).sum())
        clipped_total += clipped
        pixel_total += gray.size

    intensity_variance = float(per_frame_mean.var())
    clipping_fraction = clipped_total / pixel_total if pixel_total else 0.0
    confidence = _confidence_from_sample_size(len(frames))

    return Lighting(
        intensity_variance_over_time=intensity_variance,
        saturation_clipping_fraction=clipping_fraction,
        confidence=confidence,
    )


def _confidence_from_sample_size(n: int) -> ProbeConfidence:
    if n >= _HIGH_MIN_FRAMES:
        return ProbeConfidence.HIGH
    if n >= _MEDIUM_MIN_FRAMES:
        return ProbeConfidence.MEDIUM
    return ProbeConfidence.LOW
