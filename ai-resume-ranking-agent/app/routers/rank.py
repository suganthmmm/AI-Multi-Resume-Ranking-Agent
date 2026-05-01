"""
POST /rank
Ranks the candidates uploaded in a previous /upload call.
Uses the agent-based RAG evaluation pipeline.
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
    **Step 2 of 2 — Agent-Based Ranking**

    Provide the `session_id` returned from `/upload`.
    Optionally pass a `job_role_hint` to fine-tune the ranking focus.

    The agent will:
    1. Extract requirements from the job description
    2. For each candidate, iteratively query ChromaDB for evidence
    3. Analyze evidence using multi-step reasoning
    4. Produce a ranked list with scores, strengths, weaknesses, and explanations
    """
    # ── Validate session ─────────────────────────────────────────────────
    if not session_store.session_exists(payload.session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{payload.session_id}' not found. Please call /upload first.",
        )

    jd_text = session_store.get_job_description(payload.session_id)
    candidate_names = session_store.get_candidate_names(payload.session_id)

    if jd_text is None or candidate_names is None:
        raise HTTPException(
            status_code=500,
            detail="Session data is corrupted. Please re-upload your documents.",
        )

    # ── Run agent-based ranking pipeline ─────────────────────────────────
    logger.info(
        "Starting agent-based ranking for %d candidate(s) in session %s (role hint: %s)",
        len(candidate_names), payload.session_id, payload.job_role_hint,
    )

    result = rank_candidates(
        candidate_names=candidate_names,
        job_description=jd_text,
        session_id=payload.session_id,
        job_role_hint=payload.job_role_hint,
    )

    return result
