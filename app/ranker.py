"""
Ranking orchestrator.
Ties together scoring and reasoning to produce the final ranked list.
"""

import logging
from app.models import (
    ExtractedResume,
    ExtractedJobDescription,
    RankedCandidate,
    RankResponse,
)
from app.scorer import compute_score
from app.reasoner import generate_explanation, generate_ranking_summary

logger = logging.getLogger(__name__)


def rank_candidates(
    resumes: list[ExtractedResume],
    jd: ExtractedJobDescription,
    session_id: str,
    job_role_hint: str | None = None,
) -> RankResponse:
    """
    Full ranking pipeline:
    1. Score every candidate
    2. Sort by score descending
    3. Generate LLM explanations
    4. Build final response
    """
    if not resumes:
        return RankResponse(
            session_id=session_id,
            job_role=jd.role_title,
            total_candidates=0,
            candidates=[],
            ranking_summary="No resumes were provided for ranking.",
        )

    job_role = job_role_hint or jd.role_title

    # ── Step 1: Score ────────────────────────────────────────────────────────
    logger.info("Scoring %d candidate(s) for role: %s", len(resumes), job_role)
    scored: list[tuple[ExtractedResume, object]] = []
    for resume in resumes:
        score_breakdown = compute_score(resume, jd)
        scored.append((resume, score_breakdown))

    # ── Step 2: Sort ─────────────────────────────────────────────────────────
    scored.sort(key=lambda x: x[1].final_score, reverse=True)

    # ── Step 3: Generate explanations ────────────────────────────────────────
    ranked_candidates: list[RankedCandidate] = []
    for rank_pos, (resume, score) in enumerate(scored, start=1):
        logger.info(
            "Generating explanation for %s (rank %d, score %.1f)",
            resume.candidate_name, rank_pos, score.final_score,
        )
        strengths, weaknesses, explanation = generate_explanation(
            resume=resume,
            jd=jd,
            score=score,
            rank=rank_pos,
            total=len(scored),
        )

        ranked_candidates.append(
            RankedCandidate(
                rank=rank_pos,
                name=resume.candidate_name,
                filename=resume.filename,
                score=round(score.final_score, 1),
                score_breakdown=score,
                strengths=strengths,
                weaknesses=weaknesses,
                explanation=explanation,
            )
        )

    # ── Step 4: Summary ──────────────────────────────────────────────────────
    top_score = ranked_candidates[0].score if ranked_candidates else 0.0
    ranked_names = [c.name for c in ranked_candidates]
    summary = generate_ranking_summary(ranked_names, job_role, top_score)

    return RankResponse(
        session_id=session_id,
        job_role=job_role,
        total_candidates=len(ranked_candidates),
        candidates=ranked_candidates,
        ranking_summary=summary,
    )
