"""
AI reasoning layer.
Generates human-readable explanations, strengths, and weaknesses
for each ranked candidate using the LLM.
"""

import json
import re
import logging
from openai import OpenAI
from app.config import settings
from app.models import ExtractedResume, ExtractedJobDescription, ScoreBreakdown

logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.openai_api_key)


REASONING_SYSTEM = """You are a senior HR analyst providing candidate evaluation reports.
Given a candidate's resume data and their match scores against a job description,
produce a concise evaluation.

Respond ONLY with a valid JSON object — no explanation, no markdown fences.

JSON schema:
{
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "explanation": "<2-3 sentence paragraph explaining why the candidate is ranked at their position, covering their key differentiators and main gaps>"
}

Rules:
- strengths: 2–4 bullet points, each specific and evidence-based.
- weaknesses: 1–3 bullet points for genuine gaps (be constructive).
- explanation: must reference the rank position and key scoring factors.
- Be concise, professional, and factual. No generic filler.
"""


def _chat_json(user_prompt: str) -> dict:
    """Single LLM call; returns parsed JSON dict."""
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=600,
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)


def generate_explanation(
    resume: ExtractedResume,
    jd: ExtractedJobDescription,
    score: ScoreBreakdown,
    rank: int,
    total: int,
) -> tuple[list[str], list[str], str]:
    """
    Returns (strengths, weaknesses, explanation) for one candidate.
    Falls back to rule-based output if LLM call fails.
    """
    prompt = f"""
CANDIDATE: {resume.candidate_name}
RANK: {rank} out of {total}
FINAL SCORE: {score.final_score}/100

SCORE BREAKDOWN:
  - Skill Match:           {score.skill_match_score}/100
  - Experience Relevance:  {score.experience_relevance_score}/100
  - Project Quality:       {score.project_quality_score}/100
  - Education:             {score.education_score}/100

JOB ROLE: {jd.role_title}
REQUIRED SKILLS: {', '.join(jd.required_skills[:10])}
EXPERIENCE REQUIRED: {jd.experience_years_required} years ({jd.experience_level})

CANDIDATE DETAILS:
  Skills:         {', '.join(resume.skills[:15])}
  Experience:     {resume.experience_years} years
  Projects:       {len(resume.projects)} project(s)
  Education:      {'; '.join(resume.education[:3]) if resume.education else 'Not specified'}
  Matched Skills: {', '.join(score.matched_skills[:10])}
  Missing Skills: {', '.join(score.missing_skills[:10])}

Generate the evaluation JSON.
""".strip()

    try:
        data = _chat_json(prompt)
        strengths = data.get("strengths", [])
        weaknesses = data.get("weaknesses", [])
        explanation = data.get("explanation", "")
        if not explanation:
            raise ValueError("Empty explanation")
        return strengths, weaknesses, explanation

    except Exception as exc:
        logger.warning("LLM reasoning failed for %s: %s — using fallback.", resume.candidate_name, exc)
        return _fallback_reasoning(resume, score, rank, total, jd)


def _fallback_reasoning(
    resume: ExtractedResume,
    score: ScoreBreakdown,
    rank: int,
    total: int,
    jd: ExtractedJobDescription,
) -> tuple[list[str], list[str], str]:
    """Rule-based fallback when LLM is unavailable."""
    strengths = []
    weaknesses = []

    if score.skill_match_score >= 70:
        strengths.append(f"Strong skill alignment — matched {len(score.matched_skills)} required skills")
    if score.experience_relevance_score >= 70:
        strengths.append(f"{resume.experience_years:.0f} years of relevant experience")
    if score.project_quality_score >= 60:
        strengths.append(f"Demonstrated project experience ({len(resume.projects)} projects)")
    if score.education_score >= 70:
        strengths.append("Strong educational background")

    if score.missing_skills:
        weaknesses.append(f"Missing key skills: {', '.join(score.missing_skills[:5])}")
    if score.experience_relevance_score < 50:
        weaknesses.append("Experience level may not align with role requirements")
    if not resume.projects:
        weaknesses.append("Limited project portfolio")

    explanation = (
        f"{resume.candidate_name} is ranked #{rank} of {total} with a score of "
        f"{score.final_score:.1f}/100. "
        f"Skill match: {score.skill_match_score:.0f}/100, "
        f"Experience: {score.experience_relevance_score:.0f}/100. "
        f"{'Strong overall fit.' if score.final_score >= 70 else 'Partial fit — some gaps present.'}"
    )

    return strengths or ["Profile reviewed"], weaknesses or ["No major gaps identified"], explanation


def generate_ranking_summary(
    ranked_names: list[str],
    jd_role: str,
    top_score: float,
) -> str:
    """Generate a one-paragraph summary of the overall ranking."""
    try:
        prompt = (
            f"Write a 2-sentence executive summary of a candidate ranking for the role of '{jd_role}'. "
            f"There are {len(ranked_names)} candidates. "
            f"The top candidate is '{ranked_names[0]}' with a score of {top_score:.1f}/100. "
            f"Other candidates in order: {', '.join(ranked_names[1:])}. "
            "Be concise and professional. No JSON needed."
        )
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        return (
            f"Ranking complete for {jd_role}. "
            f"{ranked_names[0]} leads with {top_score:.1f}/100. "
            f"All {len(ranked_names)} candidates evaluated."
        )
