"""
In-memory session store.
Holds job description text and candidate names between /upload and /rank calls.
Resume content is stored in ChromaDB, not here.
"""

from typing import Optional


class SessionStore:
    """
    Simple dict-backed store keyed by session_id.
    Each session holds:
        - job_description: str (raw JD text)
        - candidate_names: list[str]
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    # ── write ──────────────────────────────────────────────────────────────

    def save(
        self,
        session_id: str,
        job_description: str,
        candidate_names: list[str],
    ) -> None:
        self._store[session_id] = {
            "job_description": job_description,
            "candidate_names": candidate_names,
        }

    # ── read ───────────────────────────────────────────────────────────────

    def get_job_description(self, session_id: str) -> Optional[str]:
        session = self._store.get(session_id)
        return session["job_description"] if session else None

    def get_candidate_names(self, session_id: str) -> Optional[list[str]]:
        session = self._store.get(session_id)
        return session["candidate_names"] if session else None

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._store

    # ── cleanup ────────────────────────────────────────────────────────────

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._store.keys())


# Singleton instance shared across the app
session_store = SessionStore()
