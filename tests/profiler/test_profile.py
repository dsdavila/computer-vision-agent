"""Profiler composition tests — glue that runs all six probes and returns
a SceneProperties. Uses a stub detector plus a synthetic MJPG video."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_agent.profiler import profile
from cv_agent.profiler.proposals import Box, FrameProposals, Proposal
from cv_agent.profiler.video import Frame, VideoReader
from cv_agent.schemas import (
    DeploymentHardware,
    HardwareEnvelope,
    Ontology,
    OperatingBias,
    OperatingPoint,
    ProbeConfidence,
    ROIPredicate,
    SceneProperties,
    ScoringMetric,
    SuccessCriterion,
    TaskContract,
)


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """A 40-frame textured video so the motion probe has real corners to
    lock onto and the pair sampler has room to spread its pairs."""
    path = tmp_path / "scene.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (128, 96))
    if not writer.isOpened():
        pytest.skip("opencv has no MJPG writer available in this build")

    rng = np.random.default_rng(0)
    base = np.full((96, 128, 3), 128, dtype=np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, 116)), int(rng.integers(0, 84))
        w, h = int(rng.integers(4, 12)), int(rng.integers(4, 12))
        base[y : y + h, x : x + w] = int(rng.integers(0, 256))

    try:
        for i in range(40):
            # Small drift per frame so motion is nonzero.
            frame = np.roll(base, shift=(0, i % 5), axis=(0, 1))
            writer.write(frame)
    finally:
        writer.release()
    return path


def _task_contract(catalog_version: str = "v0-2026-09-17") -> TaskContract:
    return TaskContract(
        name="test",
        ontology=Ontology(closed_set_classes=("person", "vehicle")),
        predicates=(
            ROIPredicate(
                name="area",
                polygon_normalized=((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)),
                target_classes=("person", "vehicle"),
            ),
        ),
        operating_point=OperatingPoint(bias=OperatingBias.BALANCED),
        deployment_hardware=DeploymentHardware(
            hardware=HardwareEnvelope(gpu_model="test", vram_gb=8, cpu_cores=4),
            latency_budget_ms=100.0,
        ),
        success_criteria=(
            SuccessCriterion(
                metric=ScoringMetric.RECALL,
                per_class="person",
                min_value=0.8,
            ),
        ),
        catalog_version=catalog_version,
    )


class StubDetector:
    """Records the targets it received and returns configured proposals per
    frame. Used to verify wiring — the profiler passes the ontology through
    correctly and consumes the detector's output for the four proposal-
    dependent probes."""

    def __init__(self, per_frame_score: float = 0.9):
        self.per_frame_score = per_frame_score
        self.calls: list[dict] = []

    def propose(
        self,
        frames: Sequence[Frame],
        targets: Sequence[str],
    ) -> Sequence[FrameProposals]:
        self.calls.append(
            {"frame_indices": [f.index for f in frames], "targets": list(targets)}
        )
        emb = np.ones(16, dtype=np.float32)
        out: list[FrameProposals] = []
        for f in frames:
            props = (
                Proposal(
                    box=Box(x1=10, y1=20, x2=40, y2=80),
                    label="person",
                    score=self.per_frame_score,
                    embedding=emb,
                ),
                Proposal(
                    box=Box(x1=50, y1=30, x2=110, y2=70),
                    label="vehicle",
                    score=self.per_frame_score - 0.05,
                    embedding=emb,
                ),
            )
            out.append(FrameProposals(frame=f, proposals=props))
        return out


class EmptyDetector:
    """Detector that finds nothing on every frame."""

    def propose(
        self,
        frames: Sequence[Frame],
        targets: Sequence[str],
    ) -> Sequence[FrameProposals]:
        del targets
        return [FrameProposals(frame=f) for f in frames]


# --- happy path -------------------------------------------------------------


def test_profile_returns_scene_properties(synthetic_video: Path):
    contract = _task_contract()
    detector = StubDetector()
    scene = profile(synthetic_video, contract, detector)
    assert isinstance(scene, SceneProperties)


def test_profile_pins_catalog_version_from_contract(synthetic_video: Path):
    contract = _task_contract(catalog_version="v0-2026-11-01")
    scene = profile(synthetic_video, contract, StubDetector())
    assert scene.catalog_version == "v0-2026-11-01"


def test_profile_passes_ontology_to_detector(synthetic_video: Path):
    contract = _task_contract()
    detector = StubDetector()
    profile(synthetic_video, contract, detector)
    # StubDetector records at least one call (one propose over the full frame
    # sample). The targets should be the ontology's target names.
    assert detector.calls
    for call in detector.calls:
        assert call["targets"] == ["person", "vehicle"]


def test_profile_accepts_video_reader_directly(synthetic_video: Path):
    contract = _task_contract()
    reader = VideoReader(synthetic_video)
    scene = profile(reader, contract, StubDetector())
    assert isinstance(scene, SceneProperties)


# --- detector output propagates -------------------------------------------


def test_high_score_detector_gives_high_target_novelty_confidence(
    synthetic_video: Path,
):
    contract = _task_contract()
    scene = profile(synthetic_video, contract, StubDetector(per_frame_score=0.9))
    assert scene.target_novelty.confidence is ProbeConfidence.HIGH
    assert scene.target_novelty.open_vocab_score_p50 >= 0.8


def test_empty_detector_yields_low_confidence_proposal_probes(
    synthetic_video: Path,
):
    contract = _task_contract()
    scene = profile(synthetic_video, contract, EmptyDetector())
    # All four proposal-dependent probes should be LOW since detector found
    # nothing.
    assert scene.pixels_on_target.confidence is ProbeConfidence.LOW
    assert scene.congestion.confidence is ProbeConfidence.LOW
    assert scene.appearance_separability.confidence is ProbeConfidence.LOW
    assert scene.target_novelty.confidence is ProbeConfidence.LOW


def test_detector_independent_probes_ignore_detector(synthetic_video: Path):
    # Detector output shouldn't affect lighting / motion readings.
    with_dets = profile(synthetic_video, _task_contract(), StubDetector())
    without_dets = profile(synthetic_video, _task_contract(), EmptyDetector())
    assert with_dets.lighting == without_dets.lighting
    assert with_dets.motion_dynamics == without_dets.motion_dynamics


# --- confidence tiers -------------------------------------------------------


def test_full_run_produces_high_confidence_lighting(synthetic_video: Path):
    # 30 frames sampled → lighting confidence should reach HIGH.
    scene = profile(synthetic_video, _task_contract(), StubDetector())
    assert scene.lighting.confidence is ProbeConfidence.HIGH


# --- error paths ------------------------------------------------------------


def test_profile_rejects_missing_video(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        profile(tmp_path / "nope.avi", _task_contract(), StubDetector())
