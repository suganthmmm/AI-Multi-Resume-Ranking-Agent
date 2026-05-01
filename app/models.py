"""
Pydantic models for request and response schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Extracted data structures ──────────────────────────────────────────────

class ExtractedResume(BaseModel):
    candidate_name: str = "Unknown Candidate"
    filename: str
    raw_text: str
    skills: list[str] = Field(default_factory=list)
    experience_years: float = 0.0
    experience_descriptions: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class ExtractedJobDescription(BaseModel):
    raw_text: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    experience_level: str = "Not specified"
    experience_years_required: float = 0.0
    key_requirements: list[str] = Field(default_factory=list)
    role_title: str = "Not specified"


# ── Scoring breakdown ──────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    skill_match_score: float = Field(..., ge=0, le=100, description="0–100 skill overlap score")
    experience_relevance_score: float = Field(..., ge=0, le=100, description="0–100 experience score")
    project_quality_score: float = Field(..., ge=0, le=100, description="0–100 project score")
    education_score: float = Field(..., ge=0, le=100, description="0–100 education score")
    final_score: float = Field(..., ge=0, le=100, description="Weighted final score")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


# ── Ranked candidate output ────────────────────────────────────────────────

class RankedCandidate(BaseModel):
    rank: int
    name: str
    filename: str
    score: float = Field(..., ge=0, le=100)
    score_breakdown: ScoreBreakdown
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    explanation: str


# ── API response models ────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    message: str
    job_description_loaded: bool
    resumes_loaded: int
    resume_filenames: list[str]
    extracted_job_role: str


class RankRequest(BaseModel):
    session_id: str
    job_role_hint: Optional[str] = Field(
        default=None,
        description="Optional hint about the job role to fine-tune ranking focus",
    )


class RankResponse(BaseModel):
    session_id: str
    job_role: str
    total_candidates: int
    candidates: list[RankedCandidate]
    ranking_summary: str
