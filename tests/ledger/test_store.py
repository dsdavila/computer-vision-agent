"""Ledger storage tests — path scheme, append-only, atomic writes,
canonical JSON, round-trip."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cv_agent.ledger import EntryAlreadyExists, LedgerStore
from cv_agent.schemas import (
    AppearanceSeparability,
    Congestion,
    CostMeasurement,
    HardwareEnvelope,
    LedgerEntry,
    Lighting,
    MetricReading,
    MotionDynamics,
    PixelsOnTarget,
    Prediction,
    ProbeConfidence,
    Provenance,
    SceneProperties,
    ScorerTier,
    TargetCostMeasurement,
    TargetNovelty,
    Verdict,
)


# --- fixtures ---------------------------------------------------------------


def _scene() -> SceneProperties:
    return SceneProperties(
        pixels_on_target=PixelsOnTarget(p10_px=18.0, p50_px=42.0, confidence=ProbeConfidence.HIGH),
        congestion=Congestion(
            detections_per_frame_mean=6.2,
            mean_pairwise_iou=0.08,
            occlusion_rate=0.15,
            confidence=ProbeConfidence.MEDIUM,
        ),
        motion_dynamics=MotionDynamics(
            optical_flow_mag_p50=1.4,
            per_track_displacement_variance=0.9,
            camera_motion_estimate=0.02,
            confidence=ProbeConfidence.HIGH,
        ),
        lighting=Lighting(
            intensity_variance_over_time=310.5,
            saturation_clipping_fraction=0.01,
            confidence=ProbeConfidence.HIGH,
        ),
        appearance_separability=AppearanceSeparability(
            inter_instance_distance_p50=0.62, confidence=ProbeConfidence.MEDIUM
        ),
        target_novelty=TargetNovelty(
            open_vocab_score_p10=0.31, open_vocab_score_p50=0.55, confidence=ProbeConfidence.LOW
        ),
        catalog_version="v0-2026-09-17",
    )


def _make_entry(entry_id: str = "entry-0001", task: str = "dock_v0") -> LedgerEntry:
    return LedgerEntry(
        id=entry_id,
        created_at=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
        parent_entry_id=None,
        task_contract_name=task,
        full_config={
            "topology.detector": "general_v1",
            "detector.confidence_threshold": 0.4,
            "detector.input_res": 640,
        },
        changed_parameters=("detector.confidence_threshold",),
        scene_properties=_scene(),
        hypothesis=(
            Prediction(
                metric_name="short_track_ratio",
                baseline_value=0.42,
                predicted_delta=-0.10,
                tolerance=0.03,
            ),
        ),
        evidence=(MetricReading(name="short_track_ratio", value=0.42),),
        outcome=(MetricReading(name="short_track_ratio", value=0.33),),
        verdict=Verdict.CONFIRMED,
        scorer_tier=ScorerTier.LABEL_FREE_PROXY,
        confounded=False,
        cost_estimated=CostMeasurement(
            config={"input_res": 640, "tile_count": 1},
            latency_ms=18.0,
            vram_mb=1400.0,
        ),
        cost_measured=TargetCostMeasurement(
            component_name="general_v1",
            catalog_version="v0-2026-09-17",
            hardware=HardwareEnvelope(gpu_model="L4", vram_gb=24, cpu_cores=16),
            config={"input_res": 640, "tile_count": 1},
            latency_ms=42.0,
            vram_mb=1900.0,
            measured_at=datetime(2026, 9, 18, tzinfo=timezone.utc),
        ),
        provenance=Provenance(
            model_id="claude-opus-4-7",
            model_version="4.7.0",
            seed=42,
            catalog_version="v0-2026-09-17",
        ),
        narrative="Raised confidence threshold; short_track_ratio fell as predicted.",
    )


# --- put + get round-trip ---------------------------------------------------


def test_put_and_get_round_trip(tmp_path: Path):
    store = LedgerStore(tmp_path)
    entry = _make_entry()
    path = store.put(entry)
    assert path == tmp_path / "dock_v0" / "entry-0001.json"
    assert path.exists()

    loaded = store.get("dock_v0", "entry-0001")
    assert loaded == entry


def test_put_creates_task_directory(tmp_path: Path):
    store = LedgerStore(tmp_path)
    store.put(_make_entry(task="brand_new"))
    assert (tmp_path / "brand_new").is_dir()


def test_json_is_pretty_and_canonical(tmp_path: Path):
    store = LedgerStore(tmp_path)
    entry = _make_entry()
    path = store.put(entry)
    text = path.read_text(encoding="utf-8")
    # Pretty-printed: multi-line JSON.
    assert "\n" in text
    # Canonical: keys sorted so diffs are review-friendly.
    payload = json.loads(text)
    assert list(payload.keys()) == sorted(payload.keys())
    # Nested dict (full_config) also sorted.
    assert list(payload["full_config"].keys()) == sorted(payload["full_config"].keys())


# --- append-only ------------------------------------------------------------


def test_put_refuses_to_overwrite(tmp_path: Path):
    store = LedgerStore(tmp_path)
    store.put(_make_entry())
    with pytest.raises(EntryAlreadyExists):
        store.put(_make_entry())


def test_put_a_second_entry_id_is_fine(tmp_path: Path):
    store = LedgerStore(tmp_path)
    store.put(_make_entry(entry_id="entry-0001"))
    store.put(_make_entry(entry_id="entry-0002"))
    assert set(store.list_entry_ids("dock_v0")) == {"entry-0001", "entry-0002"}


# --- listing / iteration ----------------------------------------------------


def test_list_tasks_returns_sorted(tmp_path: Path):
    store = LedgerStore(tmp_path)
    store.put(_make_entry(task="zeta"))
    store.put(_make_entry(task="alpha"))
    assert store.list_tasks() == ["alpha", "zeta"]


def test_list_entry_ids_sorted(tmp_path: Path):
    store = LedgerStore(tmp_path)
    store.put(_make_entry(entry_id="entry-0002"))
    store.put(_make_entry(entry_id="entry-0001"))
    assert store.list_entry_ids("dock_v0") == ["entry-0001", "entry-0002"]


def test_list_entry_ids_on_empty_task(tmp_path: Path):
    store = LedgerStore(tmp_path)
    assert store.list_entry_ids("nonexistent") == []


def test_list_tasks_on_empty_root(tmp_path: Path):
    store = LedgerStore(tmp_path / "does-not-exist-yet")
    assert store.list_tasks() == []


def test_iter_entries_yields_in_id_order(tmp_path: Path):
    store = LedgerStore(tmp_path)
    for i in [3, 1, 2]:
        store.put(_make_entry(entry_id=f"entry-{i:04d}"))
    ids = [e.id for e in store.iter_entries("dock_v0")]
    assert ids == ["entry-0001", "entry-0002", "entry-0003"]


def test_iter_all_yields_task_and_entry(tmp_path: Path):
    store = LedgerStore(tmp_path)
    store.put(_make_entry(task="alpha", entry_id="a-1"))
    store.put(_make_entry(task="beta", entry_id="b-1"))
    pairs = [(t, e.id) for t, e in store.iter_all()]
    assert pairs == [("alpha", "a-1"), ("beta", "b-1")]


def test_has(tmp_path: Path):
    store = LedgerStore(tmp_path)
    assert not store.has("dock_v0", "entry-0001")
    store.put(_make_entry())
    assert store.has("dock_v0", "entry-0001")


# --- identifier safety ------------------------------------------------------


def test_rejects_path_traversal_in_task_name(tmp_path: Path):
    store = LedgerStore(tmp_path)
    entry = _make_entry(task="dock_v0")
    # Manually build an entry with a path-traversal task; validation must fire.
    with pytest.raises(ValueError):
        store.get("../etc", "entry-0001")


def test_rejects_slashes_in_entry_id(tmp_path: Path):
    store = LedgerStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("dock_v0", "a/b")


def test_rejects_empty_identifier(tmp_path: Path):
    store = LedgerStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("", "x")
    with pytest.raises(ValueError):
        store.get("t", "")


# --- atomic-write behavior (best-effort inspection) ------------------------


def test_no_leftover_temp_files_after_successful_put(tmp_path: Path):
    store = LedgerStore(tmp_path)
    store.put(_make_entry())
    files = list((tmp_path / "dock_v0").iterdir())
    # Only the real entry.json, no dotfile temp leftovers.
    assert [f.name for f in files] == ["entry-0001.json"]
