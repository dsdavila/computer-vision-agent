"""Video I/O and frame sampling for the profiler.

Frames are numpy-backed (H x W x 3 uint8 BGR — opencv-native channel order).
Probes that need RGB call `frame.rgb()`. Everything else can work directly on
the array; statistics like intensity variance, saturation, and optical flow
are channel-order-agnostic.

VideoReader wraps opencv. It is intentionally minimal: `iter_frames()` for a
full pass and `sample_uniform(n)` for a deterministic n-frame subsample.
Random sampling requires a seed at the call site — the profiler is
deterministic (§4), so uniform is preferred where it fits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    """One decoded frame plus its position in the source video.

    `array` shape is (height, width, 3), dtype uint8, channels are BGR
    (opencv's native order). Do not mutate `array`; probes assume frames are
    read-only. Frame equality is by identity, not by pixel values."""

    array: np.ndarray
    index: int  # 0-based frame number in the source video
    timestamp_s: float  # seconds from the start of the video

    def rgb(self) -> np.ndarray:
        """Return an RGB view of the frame (for probes that consume RGB)."""
        return cv2.cvtColor(self.array, cv2.COLOR_BGR2RGB)

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])


class VideoReader:
    """Read frames from a video file. Wraps opencv."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"VideoReader: {self._path} not found")
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise IOError(f"VideoReader: opencv could not open {self._path}")
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_s(self) -> float:
        if self._fps <= 0:
            return 0.0
        return self._frame_count / self._fps

    def iter_frames(self) -> Iterator[Frame]:
        """Yield every frame in order. Opens a fresh opencv handle so callers
        can re-iterate."""
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise IOError(f"VideoReader: opencv could not open {self._path}")
        try:
            index = 0
            while True:
                ok, array = cap.read()
                if not ok:
                    break
                timestamp_s = index / self._fps if self._fps > 0 else 0.0
                yield Frame(array=array, index=index, timestamp_s=timestamp_s)
                index += 1
        finally:
            cap.release()

    def sample_uniform(self, n: int) -> list[Frame]:
        """Return n evenly-spaced frames covering the full duration.

        Deterministic: repeated calls return the same frames. n must be >= 1.
        If the video has fewer than n frames, returns all of them."""
        if n < 1:
            raise ValueError("VideoReader.sample_uniform: n must be >= 1")
        if self._frame_count <= 0:
            return []

        n = min(n, self._frame_count)
        # Evenly-spaced integer indices in [0, frame_count - 1] inclusive.
        indices = np.linspace(0, self._frame_count - 1, n).round().astype(int)
        wanted = set(indices.tolist())
        out: list[Frame] = []
        for frame in self.iter_frames():
            if frame.index in wanted:
                out.append(frame)
                if len(out) == len(wanted):
                    break
        return out


@overload
def sample_frame_pairs(reader: VideoReader, n: int) -> list[tuple[Frame, Frame]]: ...


def sample_frame_pairs(reader: VideoReader, n: int) -> list[tuple[Frame, Frame]]:
    """Return n pairs of consecutive frames evenly spaced across the video.

    Used by probes that need frame-to-frame difference (optical flow, motion
    dynamics). Each pair is (frame_at_i, frame_at_i+1). If a chosen index is
    the last frame of the video, that pair is skipped — callers see up to n
    pairs, possibly fewer near the tail."""
    if n < 1:
        raise ValueError("sample_frame_pairs: n must be >= 1")
    if reader.frame_count < 2:
        return []

    # Pick n anchors; each anchor + its successor forms a pair.
    max_anchor = reader.frame_count - 2  # last index that has a successor
    n = min(n, max_anchor + 1)
    anchors = np.linspace(0, max_anchor, n).round().astype(int)
    anchor_set = set(anchors.tolist())
    successor_set = {a + 1 for a in anchors}

    wanted = anchor_set | successor_set
    by_index: dict[int, Frame] = {}
    for frame in reader.iter_frames():
        if frame.index in wanted:
            by_index[frame.index] = frame
            if len(by_index) == len(wanted):
                break

    pairs: list[tuple[Frame, Frame]] = []
    for a in anchors.tolist():
        b = a + 1
        if a in by_index and b in by_index:
            pairs.append((by_index[a], by_index[b]))
    return pairs
