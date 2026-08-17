from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .exceptions import JobNotFoundError
from .models import JobRecord, JobState


class JobStore:
    def __init__(self, state_dir: Path):
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "jobs.sqlite3"
        with self._session() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    slurm_id TEXT,
                    state TEXT NOT NULL,
                    remote_dir TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def create(self, job_id: str, remote_dir: str, manifest: dict[str, object]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._session() as db:
            db.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, "", JobState.CREATED.value, remote_dir, json.dumps(manifest), now, now),
            )

    def update(
        self, job_id: str, *, state: JobState | None = None, slurm_id: str | None = None
    ) -> None:
        current = self.get(job_id)
        with self._session() as db:
            db.execute(
                "UPDATE jobs SET state=?, slurm_id=?, updated_at=? WHERE id=?",
                (
                    (state or JobState(current["state"])).value,
                    slurm_id if slurm_id is not None else current["slurm_id"],
                    datetime.now(timezone.utc).isoformat(),
                    job_id,
                ),
            )

    def get(self, job_id: str) -> JobRecord:
        with self._session() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        result = dict(row)
        result["manifest"] = json.loads(result["manifest"])
        return result  # type: ignore[return-value]

    def update_manifest(self, job_id: str, manifest: dict[str, object]) -> None:
        self.get(job_id)
        with self._session() as db:
            db.execute(
                "UPDATE jobs SET manifest=?, updated_at=? WHERE id=?",
                (json.dumps(manifest), datetime.now(timezone.utc).isoformat(), job_id),
            )

    def list(self) -> list[JobRecord]:
        with self._session() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        result: list[JobRecord] = []
        for row in rows:
            item = dict(row)
            item["manifest"] = json.loads(item["manifest"])
            result.append(item)  # type: ignore[arg-type]
        return result
