"""Ledger storage — files-in-git system of record (§9.1, §16 #2).

One file per LedgerEntry. Path scheme:

    <root>/<task_contract_name>/<entry_id>.json

- Append-only. `put()` refuses to overwrite an existing file; §9's
  append-only property is enforced structurally, not by convention.
- Atomic writes. Files are written to a temp file and atomically renamed
  into place, so a crash mid-write cannot leave a partial entry visible to
  readers or to git.
- Canonical JSON. Entries are serialized with sorted keys and indent 2 so
  diffs are review-friendly (the point of files-in-git; §9.1 individual
  entries are pulled on demand as backing evidence for the narrative).

Git operations are external. This layer writes files; the operator (or CI)
commits them. Keeping git out of the write path lets the store stay simple
and lets standard git workflows handle history, review, and provenance.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path

from cv_agent.schemas import LedgerEntry


_SAFE_IDENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class EntryAlreadyExists(Exception):
    """Raised when a put() would overwrite an existing entry. §9 append-only."""


class LedgerStore:
    """Filesystem-backed ledger under a root directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    # --- writes ------------------------------------------------------------

    def put(self, entry: LedgerEntry) -> Path:
        """Persist one LedgerEntry. Returns the on-disk path.

        Raises EntryAlreadyExists if an entry with the same
        (task_contract_name, id) is already stored. This is the append-only
        enforcement point."""
        path = self._path_for(entry.task_contract_name, entry.id)
        if path.exists():
            raise EntryAlreadyExists(
                f"LedgerStore: {entry.task_contract_name}/{entry.id} already exists at {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _dump_entry(entry)
        _atomic_write(path, payload)
        return path

    # --- reads -------------------------------------------------------------

    def get(self, task_contract_name: str, entry_id: str) -> LedgerEntry:
        """Load one entry by (task, id). Raises FileNotFoundError if absent."""
        path = self._path_for(task_contract_name, entry_id)
        text = path.read_text(encoding="utf-8")
        return LedgerEntry.model_validate_json(text)

    def has(self, task_contract_name: str, entry_id: str) -> bool:
        return self._path_for(task_contract_name, entry_id).exists()

    def list_tasks(self) -> list[str]:
        """Return the task_contract_name directories currently present."""
        if not self._root.exists():
            return []
        return sorted(
            d.name for d in self._root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )

    def list_entry_ids(self, task_contract_name: str) -> list[str]:
        """Return the entry ids stored under one task, sorted."""
        _validate_identifier(task_contract_name, kind="task_contract_name")
        task_dir = self._root / task_contract_name
        if not task_dir.exists():
            return []
        return sorted(p.stem for p in task_dir.iterdir() if p.suffix == ".json")

    def iter_entries(self, task_contract_name: str) -> Iterator[LedgerEntry]:
        """Yield every entry in one task, sorted by entry id."""
        for entry_id in self.list_entry_ids(task_contract_name):
            yield self.get(task_contract_name, entry_id)

    def iter_all(self) -> Iterator[tuple[str, LedgerEntry]]:
        """Yield (task_contract_name, entry) across every task and entry.

        Used by the derived-index rebuild path (§9.1) — the index is
        rebuildable from files-in-git by iterating everything the store
        contains."""
        for task in self.list_tasks():
            for entry in self.iter_entries(task):
                yield task, entry

    # --- path resolution ---------------------------------------------------

    def _path_for(self, task_contract_name: str, entry_id: str) -> Path:
        _validate_identifier(task_contract_name, kind="task_contract_name")
        _validate_identifier(entry_id, kind="entry_id")
        return self._root / task_contract_name / f"{entry_id}.json"


# --- helpers ---------------------------------------------------------------


def _validate_identifier(value: str, *, kind: str) -> None:
    if not value:
        raise ValueError(f"LedgerStore: {kind} must be non-empty")
    if not _SAFE_IDENT_RE.match(value):
        raise ValueError(
            f"LedgerStore: {kind} {value!r} must match {_SAFE_IDENT_RE.pattern}"
        )


def _dump_entry(entry: LedgerEntry) -> str:
    """Serialize LedgerEntry to canonical JSON.

    Two-step: model_dump(mode='json') to get JSON-compatible primitives
    (datetimes -> ISO strings, enums -> their values), then json.dumps with
    sort_keys=True and indent=2 for git-diff-friendly output. `full_config`
    is a dict whose insertion order is not guaranteed to be canonical across
    runs — sort_keys handles that."""
    payload = entry.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically via a same-directory temp file."""
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the temp file on failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
