from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from ..exceptions import JobNotFoundError
from .models import SessionState


class SessionRecord(TypedDict):
    id: str
    scheduler_id: str
    state: str
    remote_dir: str
    channel: str
    token: str
    manifest: dict[str, Any]
    created_at: str
    updated_at: str


class SessionStore:
    """Durable local session registry backed by the SDK SQLite database."""

    def __init__(self, state_dir: Path):
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "jobs.sqlite3"
        with self._session() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    scheduler_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    remote_dir TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    token TEXT NOT NULL,
                    manifest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

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

    def create(
        self,
        session_id: str,
        remote_dir: str,
        token: str,
        channel: str,
        manifest: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._session() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    "",
                    SessionState.CREATED.value,
                    remote_dir,
                    channel,
                    token,
                    json.dumps(manifest),
                    now,
                    now,
                ),
            )

    def update(
        self,
        session_id: str,
        *,
        state: SessionState | None = None,
        scheduler_id: str | None = None,
        channel: str | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        current = self.get(session_id)
        with self._session() as db:
            db.execute(
                """UPDATE sessions SET scheduler_id=?, state=?, channel=?, manifest=?,
                   updated_at=? WHERE id=?""",
                (
                    scheduler_id if scheduler_id is not None else current["scheduler_id"],
                    state.value if state is not None else current["state"],
                    channel if channel is not None else current["channel"],
                    json.dumps(manifest if manifest is not None else current["manifest"]),
                    datetime.now(timezone.utc).isoformat(),
                    session_id,
                ),
            )

    def get(self, session_id: str) -> SessionRecord:
        with self._session() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(session_id)
        result = dict(row)
        result["manifest"] = json.loads(result["manifest"])
        return result  # type: ignore[return-value]

    def list(self) -> list[SessionRecord]:
        with self._session() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        result: list[SessionRecord] = []
        for row in rows:
            item = dict(row)
            item["manifest"] = json.loads(item["manifest"])
            result.append(item)  # type: ignore[arg-type]
        return result
