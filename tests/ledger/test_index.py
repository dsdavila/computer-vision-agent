"""LedgerIndex tests — schema, upsert, incremental ingest, rebuild,
checkpoint, NN placeholder."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from cv_agent.ledger import LedgerIndex, LedgerStore
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


def _scene(*, p50_px: float = 42.0, catalog: str = "v0-2026-09-17") -> SceneProperties:
    return SceneProperties(
        pixels_on_target=PixelsOnTarget(
            p10_px=max(0.0, p50_px - 8.0), p50_px=p50_px, confidence=ProbeConfidence.HIGH
        ),
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
        catalog_version=catalog,
    )


def _entry(
    entry_id: str = "entry-0001",
    task: str = "dock_v0",
    *,
    p50_px: float = 42.0,
    catalog: str = "v0-2026-09-17",
    verdict: Verdict = Verdict.CONFIRMED,
) -> LedgerEntry:
    return LedgerEntry(
        id=entry_id,
        created_at=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
        parent_entry_id=None,
        task_contract_name=task,
        full_config={"detector.confidence_threshold": 0.4},
        changed_parameters=("detector.confidence_threshold",),
        scene_properties=_scene(p50_px=p50_px, catalog=catalog),
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
        verdict=verdict,
        scorer_tier=ScorerTier.LABEL_FREE_PROXY,
        confounded=False,
        cost_estimated=CostMeasurement(
            config={"input_res": 640, "tile_count": 1},
            latency_ms=18.0,
            vram_mb=1400.0,
        ),
        cost_measured=None,
        provenance=Provenance(
            model_id="claude-opus-4-7",
            model_version="4.7.0",
            seed=42,
            catalog_version=catalog,
        ),
        narrative="short_track_ratio fell as predicted.",
    )


# --- upsert + basic queries -------------------------------------------------


def test_upsert_and_has(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        assert not idx.has("dock_v0", "entry-0001")
        idx.upsert("dock_v0", _entry())
        assert idx.has("dock_v0", "entry-0001")
        assert idx.size() == 1


def test_upsert_is_idempotent(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        idx.upsert("dock_v0", _entry())
        idx.upsert("dock_v0", _entry())  # same entry twice
        assert idx.size() == 1


def test_upsert_rejects_task_mismatch(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        with pytest.raises(ValueError):
            idx.upsert("wrong_task", _entry(task="dock_v0"))


def test_list_by_task_returns_entries(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        idx.upsert("dock_v0", _entry(entry_id="entry-0002"))
        idx.upsert("dock_v0", _entry(entry_id="entry-0001"))
        idx.upsert("other", _entry(task="other", entry_id="x-1"))
        result = idx.list_by_task("dock_v0")
        assert result == [("dock_v0", "entry-0001"), ("dock_v0", "entry-0002")]


# --- refresh_from_store (incremental ingest) --------------------------------


def test_refresh_ingests_new_entries(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger")
    store.put(_entry(entry_id="entry-0001"))
    store.put(_entry(entry_id="entry-0002"))

    with LedgerIndex(tmp_path / "index.db") as idx:
        assert idx.refresh_from_store(store) == 2
        assert idx.size() == 2


def test_refresh_skips_already_indexed(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger")
    store.put(_entry(entry_id="entry-0001"))

    with LedgerIndex(tmp_path / "index.db") as idx:
        idx.refresh_from_store(store)
        # No new entries → refresh returns 0.
        assert idx.refresh_from_store(store) == 0

        # Add another entry to the store, refresh sees only the new one.
        store.put(_entry(entry_id="entry-0002"))
        assert idx.refresh_from_store(store) == 1


def test_refresh_handles_empty_store(tmp_path: Path):
    store = LedgerStore(tmp_path / "empty_ledger")
    with LedgerIndex(tmp_path / "index.db") as idx:
        assert idx.refresh_from_store(store) == 0


# --- rebuild_from_store -----------------------------------------------------


def test_rebuild_drops_and_reingests(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger")
    store.put(_entry(entry_id="entry-0001"))
    store.put(_entry(entry_id="entry-0002"))

    with LedgerIndex(tmp_path / "index.db") as idx:
        idx.refresh_from_store(store)
        assert idx.size() == 2

        # A rebuild against the same store re-ingests everything.
        assert idx.rebuild_from_store(store) == 2
        assert idx.size() == 2


def test_rebuild_preserves_checkpoints(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger")
    store.put(_entry(entry_id="entry-0001"))

    with LedgerIndex(tmp_path / "index.db") as idx:
        idx.set_checkpoint("git_sha", "abc123")
        idx.refresh_from_store(store)
        idx.rebuild_from_store(store)
        # Rebuild only drops the entry table; checkpoints survive.
        assert idx.get_checkpoint("git_sha") == "abc123"


# --- checkpoints ------------------------------------------------------------


def test_checkpoint_get_missing_returns_none(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        assert idx.get_checkpoint("git_sha") is None


def test_checkpoint_set_then_get(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        idx.set_checkpoint("git_sha", "abc123")
        assert idx.get_checkpoint("git_sha") == "abc123"


def test_checkpoint_overwrite(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        idx.set_checkpoint("git_sha", "abc123")
        idx.set_checkpoint("git_sha", "def456")
        assert idx.get_checkpoint("git_sha") == "def456"


# --- persistence across open/close -----------------------------------------


def test_persists_across_reopen(tmp_path: Path):
    db_path = tmp_path / "index.db"
    with LedgerIndex(db_path) as idx:
        idx.upsert("dock_v0", _entry())
        idx.set_checkpoint("git_sha", "abc")

    with LedgerIndex(db_path) as idx:
        assert idx.size() == 1
        assert idx.get_checkpoint("git_sha") == "abc"


# --- find_nearest (v0 placeholder) ------------------------------------------


def test_find_nearest_returns_k_sorted_by_distance(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        # Three entries at different p50_px; query at 42 -> entry_0 is closest.
        idx.upsert("dock_v0", _entry(entry_id="entry-0000", p50_px=42.0))
        idx.upsert("dock_v0", _entry(entry_id="entry-0001", p50_px=60.0))
        idx.upsert("dock_v0", _entry(entry_id="entry-0002", p50_px=100.0))

        query = _scene(p50_px=42.0)
        hits = idx.find_nearest(query, k=2)
        assert len(hits) == 2
        assert hits[0].entry_id == "entry-0000"
        assert hits[0].distance == pytest.approx(0.0)
        assert hits[1].entry_id == "entry-0001"
        assert hits[0].distance <= hits[1].distance


def test_find_nearest_respects_catalog_filter(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        idx.upsert("dock_v0", _entry(entry_id="a", catalog="v0-2026-09-17"))
        idx.upsert("dock_v0", _entry(entry_id="b", catalog="v0-2027-01-01"))

        query = _scene(catalog="v0-2026-09-17")
        hits_v0 = idx.find_nearest(query, k=10, catalog_version="v0-2026-09-17")
        assert [h.entry_id for h in hits_v0] == ["a"]

        hits_v1 = idx.find_nearest(query, k=10, catalog_version="v0-2027-01-01")
        assert [h.entry_id for h in hits_v1] == ["b"]


def test_find_nearest_rejects_k_zero(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        with pytest.raises(ValueError):
            idx.find_nearest(_scene(), k=0)


def test_find_nearest_on_empty_index_returns_empty(tmp_path: Path):
    with LedgerIndex(tmp_path / "index.db") as idx:
        assert idx.find_nearest(_scene(), k=5) == []
