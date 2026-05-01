# 🤖 AI Multi-Resume Ranking Agent

A **backend-only AI agent** built with FastAPI that analyses multiple resumes against a job description and returns a ranked list of candidates with scores and explanations — simulating how an expert HR recruiter evaluates applicants.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📄 Document parsing | PDF & DOCX via `pdfplumber` + `python-docx` |
| 🧠 AI feature extraction | GPT-powered structured extraction of skills, experience, projects, education |
| 📊 Weighted scoring | `Skills×0.4 + Experience×0.3 + Projects×0.2 + Education×0.1` |
| 🏆 Ranked output | Sorted candidates with per-component score breakdown |
| 💬 LLM explanations | Strengths, weaknesses & ranking narrative per candidate |
| ⚡ Session-based | Upload once → rank multiple times with different hints |
| 🔒 Validation | Max 5 resumes, 10 MB file limit, PDF/DOCX only |

---

## 🏗️ Architecture

```
POST /upload  →  Parse docs  →  AI extraction  →  Session store
                                                         │
POST /rank   ←  JSON response  ←  AI reasoning  ←  Scoring engine
```

```
ai-resume-ranking-agent/
├── main.py                  # FastAPI app & router wiring
├── requirements.txt
├── .env.example
├── app/
│   ├── config.py            # Settings (env vars)
│   ├── models.py            # Pydantic schemas
│   ├── store.py             # In-memory session store
│   ├── parser.py            # PDF/DOCX text extraction
│   ├── extractor.py         # LLM feature extraction
│   ├── scorer.py            # Weighted scoring engine
│   ├── reasoner.py          # LLM explanation generation
│   ├── ranker.py            # Orchestration pipeline
│   └── routers/
│       ├── upload.py        # POST /upload
│       └── rank.py          # POST /rank
└── tests/
    └── test_agent.py        # Full test suite (pytest)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ai-resume-ranking-agent.git
cd ai-resume-ranking-agent

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the Server

```bash
uvicorn main:app --reload
```

Server starts at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

---

## 📡 API Reference

### `POST /upload`

Upload a job description and up to 5 resumes.

**Request** (`multipart/form-data`):

| Field | Type | Required | Description |
|---|---|---|---|
| `job_description` | string | one of | JD as plain text |
| `job_description_file` | file | one of | JD as PDF/DOCX |
| `resumes` | file[] | ✅ | 1–5 PDF or DOCX files |

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Successfully processed 3 resume(s).",
  "job_description_loaded": true,
  "resumes_loaded": 3,
  "resume_filenames": ["alice.pdf", "bob.docx", "carol.pdf"],
  "extracted_job_role": "Senior Python Developer"
}
```

---

### `POST /rank`

Rank the candidates from an uploaded session.

**Request** (`application/json`):
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_role_hint": "backend engineer"
}
```

**Response**:
```json
{
  "session_id": "550e8400-...",
  "job_role": "Senior Python Developer",
  "total_candidates": 3,
  "ranking_summary": "Alice leads the field with strong Python and FastAPI experience.",
  "candidates": [
    {
      "rank": 1,
      "name": "Alice Smith",
      "filename": "alice.pdf",
      "score": 87.4,
      "score_breakdown": {
        "skill_match_score": 92.0,
        "experience_relevance_score": 88.0,
        "project_quality_score": 78.0,
        "education_score": 70.0,
        "final_score": 87.4,
        "matched_skills": ["Python", "FastAPI", "Docker"],
        "missing_skills": ["Kubernetes"]
      },
      "strengths": [
        "Strong skill alignment — matched 3 of 4 required skills",
        "7 years of relevant backend experience",
        "Demonstrated FastAPI and Docker projects"
      ],
      "weaknesses": ["Missing Kubernetes experience"],
      "explanation": "Alice ranks #1 of 3 with a score of 87.4/100. Her deep Python and FastAPI expertise directly matches the role requirements, and her 7 years of experience exceeds the 5-year minimum."
    }
  ]
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Health endpoints
- Document parser (clean text, section splitting, unsupported formats)
- Scoring engine (formula, range checks, skill matching)
- Session store (CRUD)
- Upload endpoint (validation, mocked AI calls)
- Rank endpoint (404 handling, schema validation, score ordering)

---

## ⚙️ Configuration

Edit `.env` to customise behaviour:

```env
OPENAI_API_KEY=sk-...          # Required
OPENAI_MODEL=gpt-4o-mini       # Or gpt-4o for best results
MAX_RESUMES=5
MAX_FILE_SIZE_MB=10

# Scoring weights (must sum to 1.0)
WEIGHT_SKILLS=0.4
WEIGHT_EXPERIENCE=0.3
WEIGHT_PROJECTS=0.2
WEIGHT_EDUCATION=0.1
```

---

## 🔧 Example cURL Usage

```bash
# Step 1 — Upload
curl -X POST http://localhost:8000/upload \
  -F "job_description=Senior Python developer needed with FastAPI Docker PostgreSQL 5+ years" \
  -F "resumes=@alice_cv.pdf" \
  -F "resumes=@bob_cv.docx"

# Step 2 — Rank (use session_id from step 1)
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID"}'
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `openai` | LLM API (extraction + reasoning) |
| `pydantic-settings` | Configuration management |
| `python-multipart` | File upload support |

---

## 📄 License

MIT
