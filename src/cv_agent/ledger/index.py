"""Derived index — §9.1 SQLite cache over the ledger.

Explicitly a cache, never authoritative. Rebuildable from the files-in-git
store by one command. §9.1 warns that git is not an index — retrieval
degrades past ~10 cases at 100–200 entries each — so a SQLite side-table
sits between the store (source of truth) and consumers that want to query
by scene properties or filter by verdict / catalog.

Sits at cost ~40K files / <100 MB at 100 problems (§9.1 back-of-envelope);
full rebuild is seconds to low minutes. Steady state uses **primary-key
diff** rather than git-log to find new entries — simpler, more reliable
than shelling to git, and independent of whether the store is actually in
git yet.

The `ingest_state` table lets whoever wants git-SHA tracking layer on top:
`set_checkpoint("git_sha", "abc123")`. The ingest itself doesn't depend on
it, so the index works even before the store is committed anywhere.

NN retrieval note: `find_nearest` is a v0 placeholder — raw Euclidean over
unnormalized scene-property features. Case-base retrieval logic is post-v0
(§10). The point of shipping this now is that the *index exists and can
rebuild*; retrieval math lands when we actually retrieve.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from cv_agent.ledger.store import LedgerStore
from cv_agent.schemas import LedgerEntry, SceneProperties


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_entries (
    task_contract_name       TEXT    NOT NULL,
    entry_id                 TEXT    NOT NULL,
    created_at               TEXT    NOT NULL,
    parent_entry_id          TEXT,
    catalog_version          TEXT    NOT NULL,
    verdict                  TEXT    NOT NULL,
    scorer_tier              INTEGER NOT NULL,
    confounded               INTEGER NOT NULL,

    pixels_p10_px            REAL    NOT NULL,
    pixels_p50_px            REAL    NOT NULL,
    pixels_confidence        TEXT    NOT NULL,

    congestion_dpfm          REAL    NOT NULL,
    congestion_iou           REAL    NOT NULL,
    congestion_occlusion     REAL    NOT NULL,
    congestion_confidence    TEXT    NOT NULL,

    motion_flow_p50          REAL    NOT NULL,
    motion_variance          REAL    NOT NULL,
    motion_camera            REAL    NOT NULL,
    motion_confidence        TEXT    NOT NULL,

    lighting_variance        REAL    NOT NULL,
    lighting_clipping        REAL    NOT NULL,
    lighting_confidence      TEXT    NOT NULL,

    separability_p50         REAL    NOT NULL,
    separability_confidence  TEXT    NOT NULL,

    novelty_p10              REAL    NOT NULL,
    novelty_p50              REAL    NOT NULL,
    novelty_confidence       TEXT    NOT NULL,

    PRIMARY KEY (task_contract_name, entry_id)
);

CREATE INDEX IF NOT EXISTS idx_ledger_task     ON ledger_entries(task_contract_name);
CREATE INDEX IF NOT EXISTS idx_ledger_catalog  ON ledger_entries(catalog_version);
CREATE INDEX IF NOT EXISTS idx_ledger_verdict  ON ledger_entries(verdict);

CREATE TABLE IF NOT EXISTS ingest_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# --- feature vector for NN queries -----------------------------------------
#
# Order matters — it's the shape of the raw feature vector for the v0
# placeholder Euclidean NN. Real case-base retrieval (post-v0) will replace
# this with a properly normalized + weighted distance.

_FEATURE_COLUMNS: tuple[str, ...] = (
    "pixels_p10_px",
    "pixels_p50_px",
    "congestion_dpfm",
    "congestion_iou",
    "congestion_occlusion",
    "motion_flow_p50",
    "motion_variance",
    "motion_camera",
    "lighting_variance",
    "lighting_clipping",
    "separability_p50",
    "novelty_p10",
    "novelty_p50",
)


def _feature_vector(scene: SceneProperties) -> tuple[float, ...]:
    return (
        scene.pixels_on_target.p10_px,
        scene.pixels_on_target.p50_px,
        scene.congestion.detections_per_frame_mean,
        scene.congestion.mean_pairwise_iou,
        scene.congestion.occlusion_rate,
        scene.motion_dynamics.optical_flow_mag_p50,
        scene.motion_dynamics.per_track_displacement_variance,
        scene.motion_dynamics.camera_motion_estimate,
        scene.lighting.intensity_variance_over_time,
        scene.lighting.saturation_clipping_fraction,
        scene.appearance_separability.inter_instance_distance_p50,
        scene.target_novelty.open_vocab_score_p10,
        scene.target_novelty.open_vocab_score_p50,
    )


@dataclass(frozen=True)
class NearestResult:
    """One hit from find_nearest. `distance` is a v0 raw Euclidean over
    unnormalized scene-property features — not a proper retrieval score."""

    task_contract_name: str
    entry_id: str
    distance: float


# --- the index --------------------------------------------------------------


class LedgerIndex:
    """SQLite-backed derived index over LedgerEntry.scene_properties.

    Open at construction, close at `.close()` or via the context manager.
    Single-writer assumption — SQLite locking handles readers alongside a
    single ingest process. Multi-writer ingest is not supported."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        # SQLite creates the file on connect if missing.
        self._conn = sqlite3.connect(self._db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> LedgerIndex:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # --- writes ------------------------------------------------------------

    def upsert(self, task_contract_name: str, entry: LedgerEntry) -> None:
        """Insert or replace one entry. Idempotent — re-ingesting the same
        entry is safe. Used by refresh() and rebuild()."""
        if entry.task_contract_name != task_contract_name:
            raise ValueError(
                f"LedgerIndex.upsert: task mismatch ({task_contract_name!r} vs "
                f"entry.task_contract_name={entry.task_contract_name!r})"
            )
        s = entry.scene_properties
        row = (
            task_contract_name,
            entry.id,
            entry.created_at.isoformat(),
            entry.parent_entry_id,
            s.catalog_version,
            entry.verdict.value,
            int(entry.scorer_tier.value),
            1 if entry.confounded else 0,

            s.pixels_on_target.p10_px,
            s.pixels_on_target.p50_px,
            s.pixels_on_target.confidence.value,

            s.congestion.detections_per_frame_mean,
            s.congestion.mean_pairwise_iou,
            s.congestion.occlusion_rate,
            s.congestion.confidence.value,

            s.motion_dynamics.optical_flow_mag_p50,
            s.motion_dynamics.per_track_displacement_variance,
            s.motion_dynamics.camera_motion_estimate,
            s.motion_dynamics.confidence.value,

            s.lighting.intensity_variance_over_time,
            s.lighting.saturation_clipping_fraction,
            s.lighting.confidence.value,

            s.appearance_separability.inter_instance_distance_p50,
            s.appearance_separability.confidence.value,

            s.target_novelty.open_vocab_score_p10,
            s.target_novelty.open_vocab_score_p50,
            s.target_novelty.confidence.value,
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO ledger_entries VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?
            )
            """,
            row,
        )
        self._conn.commit()

    def has(self, task_contract_name: str, entry_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM ledger_entries WHERE task_contract_name = ? AND entry_id = ?",
            (task_contract_name, entry_id),
        )
        return cur.fetchone() is not None

    def size(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM ledger_entries")
        return int(cur.fetchone()[0])

    def existing_keys(self) -> set[tuple[str, str]]:
        cur = self._conn.execute("SELECT task_contract_name, entry_id FROM ledger_entries")
        return {(t, e) for t, e in cur.fetchall()}

    # --- ingest from store -------------------------------------------------

    def refresh_from_store(self, store: LedgerStore) -> int:
        """Ingest anything in `store` not already in the index. §9.1
        incremental steady-state. Returns the number of entries ingested."""
        indexed = self.existing_keys()
        count = 0
        for task, entry in store.iter_all():
            if (task, entry.id) in indexed:
                continue
            self.upsert(task, entry)
            count += 1
        return count

    def rebuild_from_store(self, store: LedgerStore) -> int:
        """Drop the entry table and re-ingest everything from the store.
        §9.1 migration path (§13 catalog-version transitions). Preserves the
        ingest_state table so checkpoints survive a rebuild."""
        self._conn.execute("DELETE FROM ledger_entries")
        self._conn.commit()
        return self.refresh_from_store(store)

    # --- checkpoint (opaque to the ingest logic) ---------------------------

    def get_checkpoint(self, key: str) -> str | None:
        cur = self._conn.execute("SELECT value FROM ingest_state WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_checkpoint(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ingest_state (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    # --- queries -----------------------------------------------------------

    def list_by_task(self, task_contract_name: str) -> list[tuple[str, str]]:
        """Return (task, entry_id) tuples under one task, sorted by entry_id."""
        cur = self._conn.execute(
            """
            SELECT task_contract_name, entry_id FROM ledger_entries
            WHERE task_contract_name = ?
            ORDER BY entry_id
            """,
            (task_contract_name,),
        )
        return list(cur.fetchall())

    def find_nearest(
        self,
        query: SceneProperties,
        k: int = 10,
        *,
        catalog_version: str | None = None,
    ) -> list[NearestResult]:
        """v0 placeholder: raw Euclidean over unnormalized scene-property
        features. Case-base retrieval (§10) is post-v0; when it lands it
        will replace this with a normalized + weighted distance.

        Filter by catalog_version if provided — retrieval across catalog
        versions produces silently-wrong nearest neighbours unless you know
        the axes stayed comparable across the version bump."""
        if k < 1:
            raise ValueError("find_nearest: k must be >= 1")

        sql = (
            "SELECT task_contract_name, entry_id, "
            + ", ".join(_FEATURE_COLUMNS)
            + " FROM ledger_entries"
        )
        params: tuple = ()
        if catalog_version is not None:
            sql += " WHERE catalog_version = ?"
            params = (catalog_version,)

        cur = self._conn.execute(sql, params)
        q_vec = _feature_vector(query)

        scored: list[tuple[float, str, str]] = []
        for row in cur:
            task, entry_id = row[0], row[1]
            features = row[2:]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(features, q_vec)))
            scored.append((dist, task, entry_id))

        scored.sort()
        return [
            NearestResult(task_contract_name=t, entry_id=e, distance=d)
            for d, t, e in scored[:k]
        ]

    # --- introspection -----------------------------------------------------

    def iter_indexed_keys(self) -> Iterator[tuple[str, str]]:
        cur = self._conn.execute(
            "SELECT task_contract_name, entry_id FROM ledger_entries ORDER BY task_contract_name, entry_id"
        )
        yield from cur.fetchall()
