"""
Ranking orchestrator (Agent-Based).
Uses the autonomous agent system to evaluate each candidate
via RAG retrieval and multi-step reasoning, then ranks them.
"""

import logging
from app.models import RankedCandidate, RankResponse
from app.agent import extract_requirements, evaluate_all_candidates

logger = logging.getLogger(__name__)


def rank_candidates(
    candidate_names: list[str],
    job_description: str,
    session_id: str,
    job_role_hint: str | None = None,
) -> RankResponse:
    """
    Full agent-based ranking pipeline:
    1. Extract requirements from JD using LLM
    2. Evaluate ALL candidates together (single LLM call):
       - Query ChromaDB for each requirement
       - Analyze evidence
       - Score based on evidence
    3. Sort candidates (descending)
    4. Assign ranks

    This does NOT use single-pass scoring.
    """
    if not candidate_names:
        return RankResponse(
            session_id=session_id,
            job_role="Unknown",
            total_candidates=0,
            candidates=[],
            ranking_summary="No resumes were provided for ranking.",
        )

    # ── Step 1: Extract requirements from JD ─────────────────────────────
    logger.info("=" * 70)
    logger.info("STARTING AGENT-BASED RANKING PIPELINE")
    logger.info("Session: %s | Candidates: %d", session_id, len(candidate_names))
    logger.info("=" * 70)

    req_data = extract_requirements(job_description)
    requirements = req_data.get("requirements", [])
    role_title = job_role_hint or req_data.get("role_title", "Unknown Role")

    logger.info("Role: %s | Requirements extracted: %d", role_title, len(requirements))

    # ── Step 2: Evaluate ALL candidates in batch (minimizes API calls) ───
    candidate_results = evaluate_all_candidates(
        candidate_names=candidate_names,
        session_id=session_id,
        requirements=requirements,
        job_description=job_description,
    )

    # ── Step 3: Sort by score descending ─────────────────────────────────
    candidate_results.sort(key=lambda x: x["score"], reverse=True)

    # ── Step 4: Assign ranks and build response ──────────────────────────
    ranked_candidates = []
    for rank_pos, result in enumerate(candidate_results, start=1):
        logger.info(
            "Rank #%d: %s (score: %.1f)",
            rank_pos, result["name"], result["score"],
        )

        ranked_candidates.append(
            RankedCandidate(
                rank=rank_pos,
                name=result["name"],
                score=round(result["score"], 1),
                strengths=result.get("strengths", []),
                weaknesses=result.get("weaknesses", []),
                evidence=result.get("evidence", []),
                explanation=result.get("explanation", ""),
            )
        )

    # ── Step 5: Generate ranking summary ─────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("RANKING COMPLETE")
    logger.info("=" * 70)

    top_name = ranked_candidates[0].name if ranked_candidates else "N/A"
    top_score = ranked_candidates[0].score if ranked_candidates else 0

    summary = (
        f"Ranking complete for '{role_title}'. "
        f"{top_name} leads with {top_score}/100. "
        f"All {len(ranked_candidates)} candidates evaluated using "
        f"RAG-based evidence retrieval and multi-step agent reasoning "
        f"across {len(requirements)} job requirements."
    )

    return RankResponse(
        session_id=session_id,
        job_role=role_title,
        total_candidates=len(ranked_candidates),
        candidates=ranked_candidates,
        ranking_summary=summary,
    )
