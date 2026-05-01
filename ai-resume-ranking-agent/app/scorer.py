"""
Scoring engine.
Computes weighted scores for each resume against the job description.

Score = (Skill Match × 0.4) + (Experience Relevance × 0.3)
      + (Project Quality × 0.2) + (Education × 0.1)

All sub-scores are normalised to 0–100 before weighting.
"""

import re
import logging
from app.models import ExtractedResume, ExtractedJobDescription, ScoreBreakdown
from app.config import settings

logger = logging.getLogger(__name__)


# ── Utilities ────────────────────────────────────────────────────────────────

def _normalise(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp a value to [0, 100]."""
    return round(max(min_val, min(max_val, value)), 2)


def _skill_tokens(skill: str) -> set[str]:
    """Lower-case token set for fuzzy skill matching."""
    return set(re.findall(r"\w+", skill.lower()))


def _skills_match(resume_skills: list[str], jd_skills: list[str]) -> tuple[float, list[str], list[str]]:
    """
    Returns:
        overlap_ratio (0–1), matched_skills list, missing_skills list
    """
    if not jd_skills:
        return 0.5, [], []  # no baseline → neutral score

    matched, missing = [], []
    for jd_skill in jd_skills:
        jd_tokens = _skill_tokens(jd_skill)
        found = any(
            len(jd_tokens & _skill_tokens(r_skill)) >= max(1, len(jd_tokens) // 2)
            for r_skill in resume_skills
        )
        if found:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    ratio = len(matched) / len(jd_skills)
    return ratio, matched, missing


# ── Sub-score calculators ────────────────────────────────────────────────────

def _score_skills(resume: ExtractedResume, jd: ExtractedJobDescription) -> tuple[float, list[str], list[str]]:
    """Skill overlap: required skills weighted higher than preferred."""
    req_ratio, req_matched, req_missing = _skills_match(resume.skills, jd.required_skills)
    pref_ratio, pref_matched, _ = _skills_match(resume.skills, jd.preferred_skills)

    if jd.preferred_skills:
        combined = req_ratio * 0.7 + pref_ratio * 0.3
    else:
        combined = req_ratio

    matched_all = list(set(req_matched + pref_matched))
    return _normalise(combined * 100), matched_all, req_missing


def _score_experience(resume: ExtractedResume, jd: ExtractedJobDescription) -> float:
    """Experience relevance based on years and level."""
    required = jd.experience_years_required
    actual = resume.experience_years

    # Years-based score
    if required == 0:
        years_score = 70.0  # unknown requirement → mid score
    elif actual >= required:
        # Reward up to 1.5× required, then cap
        years_score = min(100.0, 70.0 + 30.0 * min(1.0, (actual - required) / max(required, 1)))
    else:
        years_score = max(0.0, (actual / required) * 70.0)

    # Level alignment bonus
    level = jd.experience_level.lower()
    level_bonus = 0.0
    if "junior" in level or "entry" in level:
        level_bonus = 10.0 if actual <= 2 else 0.0
    elif "mid" in level or "intermediate" in level:
        level_bonus = 10.0 if 2 < actual <= 5 else 0.0
    elif "senior" in level or "lead" in level or "principal" in level:
        level_bonus = 10.0 if actual > 5 else 0.0

    return _normalise(years_score + level_bonus)


def _score_projects(resume: ExtractedResume, jd: ExtractedJobDescription) -> float:
    """Project relevance: check if project descriptions mention JD keywords."""
    if not resume.projects:
        return 20.0  # some baseline

    all_skills = set(s.lower() for s in jd.required_skills + jd.preferred_skills)
    if not all_skills:
        # Fall back: give partial credit for having projects
        return min(100.0, 40.0 + len(resume.projects) * 10.0)

    project_text = " ".join(resume.projects).lower()
    hits = sum(1 for skill in all_skills if re.search(r"\b" + re.escape(skill) + r"\b", project_text))
    ratio = hits / len(all_skills) if all_skills else 0

    base = 30.0 + ratio * 60.0
    # Bonus for having more projects (up to 5)
    base += min(len(resume.projects), 5) * 2
    return _normalise(base)


def _score_education(resume: ExtractedResume, jd: ExtractedJobDescription) -> float:
    """Education score based on degree level keywords."""
    if not resume.education:
        return 30.0

    edu_text = " ".join(resume.education).lower()

    if any(k in edu_text for k in ("phd", "doctorate", "ph.d")):
        return 100.0
    if any(k in edu_text for k in ("master", "msc", "m.sc", "mba", "m.eng")):
        return 85.0
    if any(k in edu_text for k in ("bachelor", "bsc", "b.sc", "b.e", "b.tech", "b.eng", "degree")):
        return 70.0
    if any(k in edu_text for k in ("diploma", "associate", "higher national")):
        return 55.0
    if any(k in edu_text for k in ("certification", "bootcamp", "course")):
        return 45.0

    return 40.0  # education mentioned but unclear


# ── Main scoring function ────────────────────────────────────────────────────

def compute_score(resume: ExtractedResume, jd: ExtractedJobDescription) -> ScoreBreakdown:
    """
    Compute the weighted score for a single candidate.

    Weights (from config):
        skills      = 0.40
        experience  = 0.30
        projects    = 0.20
        education   = 0.10
    """
    skill_score, matched_skills, missing_skills = _score_skills(resume, jd)
    exp_score = _score_experience(resume, jd)
    proj_score = _score_projects(resume, jd)
    edu_score = _score_education(resume, jd)

    final = (
        skill_score * settings.weight_skills
        + exp_score * settings.weight_experience
        + proj_score * settings.weight_projects
        + edu_score * settings.weight_education
    )

    return ScoreBreakdown(
        skill_match_score=skill_score,
        experience_relevance_score=exp_score,
        project_quality_score=proj_score,
        education_score=edu_score,
        final_score=_normalise(final),
        matched_skills=matched_skills,
        missing_skills=missing_skills,
    )
