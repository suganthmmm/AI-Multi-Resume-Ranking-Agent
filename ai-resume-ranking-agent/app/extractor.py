"""
AI-powered feature extraction.
Uses the Gemini API to pull structured data from raw resume / JD text.
"""

import json
import logging
import re
import google.generativeai as genai
from app.config import settings
from app.models import ExtractedResume, ExtractedJobDescription

logger = logging.getLogger(__name__)
genai.configure(api_key=settings.gemini_api_key)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _chat(system: str, user: str, max_tokens: int = 1500) -> str:
    """Single-turn chat completion. Returns the assistant message text."""
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
    )
    response = model.generate_content(user)
    return response.text.strip()


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


# ── Resume extraction ────────────────────────────────────────────────────────

RESUME_SYSTEM = """You are an expert HR data parser.
Extract structured information from the resume text provided.
Respond ONLY with a valid JSON object — no explanation, no markdown fences.

JSON schema:
{
  "candidate_name": "<full name or 'Unknown Candidate'>",
  "skills": ["<skill1>", "<skill2>", ...],
  "experience_years": <number>,
  "experience_descriptions": ["<one sentence per role>", ...],
  "projects": ["<brief project description>", ...],
  "education": ["<degree, institution, year>", ...],
  "certifications": ["<certification name>", ...]
}
Rules:
- skills: list every technical and soft skill mentioned.
- experience_years: total years of professional experience (0 if student/fresher).
- experience_descriptions: one sentence per job role summarising responsibilities.
- projects: one sentence per project.
- education: include degree name, institution, graduation year if available.
- Use empty lists [] when data is absent; never return null.
"""


def extract_resume_features(text: str, filename: str) -> ExtractedResume:
    """Call LLM to extract structured fields from a single resume."""
    try:
        raw = _chat(RESUME_SYSTEM, f"RESUME TEXT:\n{text[:6000]}")
        data = _parse_json(raw)
    except Exception as exc:
        logger.warning("Resume extraction failed for %s: %s", filename, exc)
        data = {}

    return ExtractedResume(
        candidate_name=data.get("candidate_name", "Unknown Candidate"),
        filename=filename,
        raw_text=text,
        skills=data.get("skills", []),
        experience_years=float(data.get("experience_years", 0) or 0),
        experience_descriptions=data.get("experience_descriptions", []),
        projects=data.get("projects", []),
        education=data.get("education", []),
        certifications=data.get("certifications", []),
    )


# ── Job Description extraction ───────────────────────────────────────────────

JD_SYSTEM = """You are an expert HR analyst.
Extract structured information from the job description provided.
Respond ONLY with a valid JSON object — no explanation, no markdown fences.

JSON schema:
{
  "role_title": "<job title>",
  "required_skills": ["<skill>", ...],
  "preferred_skills": ["<skill>", ...],
  "experience_level": "<junior|mid|senior|lead|not specified>",
  "experience_years_required": <number>,
  "key_requirements": ["<one sentence>", ...]
}
Rules:
- required_skills: must-have skills explicitly stated.
- preferred_skills: nice-to-have or bonus skills.
- experience_level: infer from context if not explicit.
- experience_years_required: minimum years stated (0 if not mentioned).
- key_requirements: up to 8 most important hiring criteria.
- Use empty lists [] when absent; never return null.
"""


def extract_jd_features(text: str) -> ExtractedJobDescription:
    """Call LLM to extract structured fields from a job description."""
    try:
        raw = _chat(JD_SYSTEM, f"JOB DESCRIPTION:\n{text[:4000]}")
        data = _parse_json(raw)
    except Exception as exc:
        logger.warning("JD extraction failed: %s", exc)
        data = {}

    return ExtractedJobDescription(
        raw_text=text,
        required_skills=data.get("required_skills", []),
        preferred_skills=data.get("preferred_skills", []),
        experience_level=data.get("experience_level", "Not specified"),
        experience_years_required=float(data.get("experience_years_required", 0) or 0),
        key_requirements=data.get("key_requirements", []),
        role_title=data.get("role_title", "Not specified"),
    )
