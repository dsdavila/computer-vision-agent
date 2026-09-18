"""Video I/O + frame-sampling tests. Uses a small synthetic video written to
a tmp directory so the tests are self-contained."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_agent.profiler.video import Frame, VideoReader, sample_frame_pairs


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Write a 30-frame 64x48 AVI to tmp_path using MJPG (the codec bundled
    with opencv-python-headless). Each frame is a solid grey linearly ramping
    from black to white, so frame index is recoverable from pixel intensity."""
    path = tmp_path / "synthetic.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("opencv has no MJPG writer available in this build")
    try:
        for i in range(30):
            gray = int(round(i * 255 / 29))
            frame = np.full((48, 64, 3), gray, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_video_reader_opens_and_reports_metadata(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    assert reader.frame_count == 30
    assert reader.fps == pytest.approx(30.0)
    assert reader.duration_s == pytest.approx(1.0)


def test_video_reader_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "does_not_exist.mp4")


def test_iter_frames_returns_all_in_order(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    frames = list(reader.iter_frames())
    assert len(frames) == 30
    assert [f.index for f in frames] == list(range(30))
    for f in frames:
        assert isinstance(f, Frame)
        assert f.array.shape == (48, 64, 3)
        assert f.array.dtype == np.uint8


def test_iter_frames_timestamps_monotonic(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    ts = [f.timestamp_s for f in reader.iter_frames()]
    assert ts == sorted(ts)
    assert ts[0] == pytest.approx(0.0)


def test_sample_uniform_returns_evenly_spaced(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    sample = reader.sample_uniform(5)
    assert len(sample) == 5
    indices = [f.index for f in sample]
    # 5 evenly-spaced indices over [0, 29].
    # 5 evenly-spaced anchors: linspace(0, 29, 5).round() → [0, 7, 14/15, 22, 29].
    # Numpy's banker's rounding sends 14.5 to 14, so we tolerate either.
    assert indices[0] == 0 and indices[-1] == 29
    assert 6 <= indices[1] <= 8
    assert 14 <= indices[2] <= 16
    assert 21 <= indices[3] <= 23


def test_sample_uniform_is_deterministic(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    a = [f.index for f in reader.sample_uniform(6)]
    b = [f.index for f in reader.sample_uniform(6)]
    assert a == b


def test_sample_uniform_over_available_returns_all(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    sample = reader.sample_uniform(999)
    assert len(sample) == 30


def test_sample_uniform_rejects_zero(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    with pytest.raises(ValueError):
        reader.sample_uniform(0)


def test_sample_frame_pairs_returns_consecutive_pairs(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    pairs = sample_frame_pairs(reader, 4)
    assert len(pairs) == 4
    for a, b in pairs:
        assert b.index == a.index + 1


def test_sample_frame_pairs_deterministic(synthetic_video: Path):
    reader = VideoReader(synthetic_video)
    a = [(p[0].index, p[1].index) for p in sample_frame_pairs(reader, 5)]
    b = [(p[0].index, p[1].index) for p in sample_frame_pairs(reader, 5)]
    assert a == b


def test_frame_rgb_swaps_channels():
    # Not tied to a real file — just check the color-convert path.
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue channel in BGR
    frame = Frame(array=bgr, index=0, timestamp_s=0.0)
    rgb = frame.rgb()
    assert rgb.shape == (4, 4, 3)
    # After BGR->RGB, the blue plane moves to channel 2, and channel 0 is 0.
    assert np.all(rgb[..., 0] == 0)
    assert np.all(rgb[..., 2] == 255)


def test_frame_dimensions():
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = Frame(array=bgr, index=3, timestamp_s=0.1)
    assert frame.width == 64
    assert frame.height == 48
