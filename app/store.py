"""
In-memory session store.
Holds parsed resumes and job descriptions between /upload and /rank calls.
"""

from typing import Optional
from app.models import ExtractedResume, ExtractedJobDescription


class SessionStore:
    """
    Simple dict-backed store keyed by session_id.
    Each session holds:
        - job_description: ExtractedJobDescription
        - resumes: list[ExtractedResume]
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    # ── write ──────────────────────────────────────────────────────────────

    def save(
        self,
        session_id: str,
        job_description: ExtractedJobDescription,
        resumes: list[ExtractedResume],
    ) -> None:
        self._store[session_id] = {
            "job_description": job_description,
            "resumes": resumes,
        }

    # ── read ───────────────────────────────────────────────────────────────

    def get_job_description(self, session_id: str) -> Optional[ExtractedJobDescription]:
        session = self._store.get(session_id)
        return session["job_description"] if session else None

    def get_resumes(self, session_id: str) -> Optional[list[ExtractedResume]]:
        session = self._store.get(session_id)
        return session["resumes"] if session else None

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._store

    # ── cleanup ────────────────────────────────────────────────────────────

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._store.keys())


# Singleton instance shared across the app
session_store = SessionStore()
