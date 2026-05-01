"""
Test suite for the AI Multi-Resume Ranking Agent.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # compact tracebacks
"""

import io
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app
from app.store import session_store
from app.models import (
    ExtractedResume,
    ExtractedJobDescription,
    ScoreBreakdown,
)
from app.scorer import compute_score
from app.parser import clean_text, split_into_sections


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_jd() -> ExtractedJobDescription:
    return ExtractedJobDescription(
        raw_text="Senior Python Developer with FastAPI and Docker experience.",
        required_skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
        preferred_skills=["Redis", "Kubernetes"],
        experience_level="senior",
        experience_years_required=5.0,
        key_requirements=["Backend API development", "Database design"],
        role_title="Senior Python Developer",
    )


@pytest.fixture
def strong_resume() -> ExtractedResume:
    return ExtractedResume(
        candidate_name="Alice Smith",
        filename="alice_smith.pdf",
        raw_text="Alice Smith — Senior Python Developer",
        skills=["Python", "FastAPI", "Docker", "PostgreSQL", "Redis", "AWS"],
        experience_years=7.0,
        experience_descriptions=["Led backend team at TechCorp (3 yrs)", "API developer at StartupX (4 yrs)"],
        projects=["Built REST API with FastAPI and Docker", "Microservices with PostgreSQL"],
        education=["Bachelor of Computer Science, MIT, 2017"],
        certifications=["AWS Certified Developer"],
    )


@pytest.fixture
def weak_resume() -> ExtractedResume:
    return ExtractedResume(
        candidate_name="Bob Jones",
        filename="bob_jones.pdf",
        raw_text="Bob Jones — Junior Web Developer",
        skills=["HTML", "CSS", "JavaScript"],
        experience_years=1.0,
        experience_descriptions=["Intern at WebAgency (1 yr)"],
        projects=["Built a personal blog"],
        education=["High school diploma"],
        certifications=[],
    )


# ── Health endpoints ──────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "AI Multi-Resume Ranking Agent"
        assert data["status"] == "running"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:
    def test_clean_text_removes_extra_spaces(self):
        raw = "Hello   World\n\n\n\nEnd"
        result = clean_text(raw)
        assert "   " not in result
        assert result.count("\n") <= 2

    def test_clean_text_strips(self):
        assert clean_text("  hello  ") == "hello"

    def test_split_into_sections_returns_dict(self):
        text = "John Doe\n\nSkills\nPython Django\n\nExperience\n5 years at TechCorp"
        sections = split_into_sections(text)
        assert isinstance(sections, dict)

    def test_unsupported_file_raises(self):
        from app.parser import extract_text_from_file
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text_from_file(b"data", "file.txt")


# ── Scorer tests ──────────────────────────────────────────────────────────────

class TestScorer:
    def test_strong_candidate_scores_higher(self, strong_resume, weak_resume, sample_jd):
        strong_score = compute_score(strong_resume, sample_jd)
        weak_score = compute_score(weak_resume, sample_jd)
        assert strong_score.final_score > weak_score.final_score

    def test_score_in_range(self, strong_resume, sample_jd):
        score = compute_score(strong_resume, sample_jd)
        assert 0 <= score.final_score <= 100
        assert 0 <= score.skill_match_score <= 100
        assert 0 <= score.experience_relevance_score <= 100
        assert 0 <= score.project_quality_score <= 100
        assert 0 <= score.education_score <= 100

    def test_matched_skills_subset_of_required(self, strong_resume, sample_jd):
        score = compute_score(strong_resume, sample_jd)
        for skill in score.matched_skills:
            assert skill in sample_jd.required_skills + sample_jd.preferred_skills

    def test_missing_skills_detected(self, weak_resume, sample_jd):
        score = compute_score(weak_resume, sample_jd)
        assert len(score.missing_skills) > 0

    def test_weighted_formula(self, strong_resume, sample_jd):
        from app.config import settings
        score = compute_score(strong_resume, sample_jd)
        expected = (
            score.skill_match_score * settings.weight_skills
            + score.experience_relevance_score * settings.weight_experience
            + score.project_quality_score * settings.weight_projects
            + score.education_score * settings.weight_education
        )
        assert abs(score.final_score - round(expected, 2)) < 0.1

    def test_empty_skills_resume(self, sample_jd):
        resume = ExtractedResume(
            candidate_name="Empty", filename="empty.pdf", raw_text="",
            skills=[], experience_years=0, experience_descriptions=[],
            projects=[], education=[], certifications=[],
        )
        score = compute_score(resume, sample_jd)
        assert 0 <= score.final_score <= 100


# ── Session store tests ───────────────────────────────────────────────────────

class TestSessionStore:
    def test_save_and_retrieve(self):
        session_store.save("test-123", "Raw JD Text", ["Alice Smith"])
        assert session_store.session_exists("test-123")
        assert session_store.get_job_description("test-123") == "Raw JD Text"
        assert len(session_store.get_candidate_names("test-123")) == 1
        session_store.delete("test-123")

    def test_missing_session(self):
        assert not session_store.session_exists("nonexistent-id")
        assert session_store.get_job_description("nonexistent-id") is None
        assert session_store.get_candidate_names("nonexistent-id") is None

    def test_delete(self):
        session_store.save("del-test", "Raw JD Text", ["Alice Smith"])
        session_store.delete("del-test")
        assert not session_store.session_exists("del-test")


# ── Upload endpoint tests ─────────────────────────────────────────────────────

class TestUploadEndpoint:
    def _make_pdf(self) -> bytes:
        """Minimal valid PDF bytes."""
        return (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
        )

    def test_missing_jd_returns_400(self, client):
        pdf = self._make_pdf()
        r = client.post(
            "/upload",
            data={"job_description": ""},
            files=[("resumes", ("cv.pdf", io.BytesIO(pdf), "application/pdf"))],
        )
        assert r.status_code == 400

    def test_missing_resumes_returns_422_or_400(self, client):
        r = client.post(
            "/upload",
            data={"job_description": "We need a Python developer."},
        )
        assert r.status_code in (400, 422)

    def test_too_many_resumes_returns_400(self, client):
        pdf = self._make_pdf()
        files = [("resumes", (f"cv{i}.pdf", io.BytesIO(pdf), "application/pdf")) for i in range(6)]
        r = client.post(
            "/upload",
            data={"job_description": "Need a Python developer"},
            files=files,
        )
        assert r.status_code == 400
        assert "Maximum" in r.json()["detail"]

    def test_unsupported_file_type_returns_415(self, client):
        r = client.post(
            "/upload",
            data={"job_description": "Need a Python developer"},
            files=[("resumes", ("cv.txt", io.BytesIO(b"text"), "text/plain"))],
        )
        assert r.status_code == 415

    @patch("app.routers.upload.add_documents")
    @patch("app.routers.upload.extract_text_from_file")
    def test_successful_upload(self, mock_parse, mock_add_documents, client):
        mock_parse.return_value = "resume text"
        mock_add_documents.return_value = 2

        pdf = self._make_pdf()
        r = client.post(
            "/upload",
            data={"job_description": "We need a Python developer with 3 years experience."},
            files=[("resumes", ("alice.pdf", io.BytesIO(pdf), "application/pdf"))],
        )
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["resumes_loaded"] == 1
        assert data["job_description_loaded"] is True


# ── Rank endpoint tests ───────────────────────────────────────────────────────

class TestRankEndpoint:
    def test_invalid_session_returns_404(self, client):
        r = client.post("/rank", json={"session_id": "does-not-exist"})
        assert r.status_code == 404

    @patch("app.routers.rank.rank_candidates")
    def test_successful_ranking(self, mock_rank, client, strong_resume, sample_jd):
        from app.models import RankResponse, RankedCandidate, ScoreBreakdown

        session_store.save("rank-test-1", sample_jd, [strong_resume])

        mock_score = ScoreBreakdown(
            skill_match_score=90, experience_relevance_score=85,
            project_quality_score=80, education_score=75,
            final_score=86.0, matched_skills=["Python"], missing_skills=[],
        )
        mock_rank.return_value = RankResponse(
            session_id="rank-test-1",
            job_role="Senior Python Developer",
            total_candidates=1,
            candidates=[
                RankedCandidate(
                    rank=1, name="Alice Smith", filename="alice_smith.pdf",
                    score=86.0,
                    strengths=["Strong Python skills"],
                    weaknesses=[], explanation="Alice ranks #1 due to strong skill match.",
                )
            ],
            ranking_summary="Alice leads the field.",
        )

        r = client.post("/rank", json={"session_id": "rank-test-1"})
        assert r.status_code == 200
        data = r.json()
        assert data["total_candidates"] == 1
        assert data["candidates"][0]["rank"] == 1
        assert data["candidates"][0]["score"] == 86.0

        session_store.delete("rank-test-1")

    def test_rank_response_schema(self, client):
        """Integration: verify full response schema without mocking ranker."""
        session_store.save("schema-test", "JD text", ["Alice Smith", "Bob Jones"])

        with patch("app.ranker.evaluate_candidate") as mock_eval:
            mock_eval.side_effect = [
                {"name": "Alice Smith", "score": 85.0, "strengths": ["A"], "weaknesses": [], "evidence": [], "explanation": "A is good", "requirement_evaluations": []},
                {"name": "Bob Jones", "score": 60.0, "strengths": [], "weaknesses": ["B"], "evidence": [], "explanation": "B is ok", "requirement_evaluations": []}
            ]
            r = client.post("/rank", json={"session_id": "schema-test"})

        assert r.status_code == 200
        data = r.json()
        assert "candidates" in data
        assert len(data["candidates"]) == 2
        # Must be sorted by rank
        ranks = [c["rank"] for c in data["candidates"]]
        assert ranks == sorted(ranks)
        # Scores must be descending
        scores = [c["score"] for c in data["candidates"]]
        assert scores == sorted(scores, reverse=True)

        for candidate in data["candidates"]:
            assert "name" in candidate
            assert "score" in candidate
            assert "strengths" in candidate
            assert "weaknesses" in candidate
            assert "explanation" in candidate

        session_store.delete("schema-test")
