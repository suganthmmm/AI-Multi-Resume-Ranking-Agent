"""
POST /rank
Ranks the candidates uploaded in a previous /upload call.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models import RankRequest, RankResponse
from app.store import session_store
from app.ranker import rank_candidates

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=RankResponse, summary="Rank uploaded candidates")
async def rank_resumes(payload: RankRequest) -> RankResponse:
    """
    **Step 2 of 2 — Rank**

    Provide the `session_id` returned from `/upload`.
    Optionally pass a `job_role_hint` to fine-tune the ranking focus.

    Returns a ranked list of candidates with scores, strengths, weaknesses,
    and an AI-generated explanation for each placement.
    """
    # ── Validate session ─────────────────────────────────────────────────────
    if not session_store.session_exists(payload.session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{payload.session_id}' not found. Please call /upload first.",
        )

    jd = session_store.get_job_description(payload.session_id)
    resumes = session_store.get_resumes(payload.session_id)

    if jd is None or resumes is None:
        raise HTTPException(
            status_code=500,
            detail="Session data is corrupted. Please re-upload your documents.",
        )

    # ── Run ranking pipeline ─────────────────────────────────────────────────
    logger.info(
        "Ranking %d candidate(s) for session %s (role hint: %s)",
        len(resumes), payload.session_id, payload.job_role_hint,
    )

    result = rank_candidates(
        resumes=resumes,
        jd=jd,
        session_id=payload.session_id,
        job_role_hint=payload.job_role_hint,
    )

    return result
