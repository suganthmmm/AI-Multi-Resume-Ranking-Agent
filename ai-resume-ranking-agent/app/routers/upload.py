"""
POST /upload
Accepts a job description (text or file) plus up to 5 resume files.
Processes: Extract text → Chunk → Embed → Store in ChromaDB.
Stores minimal metadata (candidate names, JD text) in memory.
"""

import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models import UploadResponse
from app.parser import extract_text_from_file, chunk_text
from app.chroma_store import add_documents
from app.store import session_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=UploadResponse, summary="Upload job description and resumes")
async def upload_documents(
    job_description: str = Form(
        default="",
        description="Job description as plain text. Leave empty if uploading a file.",
    ),
    job_description_file: UploadFile | None = File(
        default=None,
        description="Job description as a PDF or DOCX file (optional).",
    ),
    resumes: list[UploadFile] = File(
        description="Resume files (PDF or DOCX). Maximum 5 files.",
    ),
) -> UploadResponse:
    """
    **Step 1 of 2 — Upload & Ingest**

    Pipeline: Extract text → Chunk (500 chars, 50 overlap) → Embed → Store in ChromaDB.

    - Supply the job description as plain text **or** a file (PDF/DOCX).
    - Upload 1–5 resume files (PDF/DOCX).
    - Returns a `session_id` to use in the subsequent `/rank` call.
    """

    # ── 1. Validate resumes count ────────────────────────────────────────
    if not resumes or (len(resumes) == 1 and not resumes[0].filename):
        raise HTTPException(status_code=400, detail="At least one resume file is required.")
    if len(resumes) > settings.max_resumes:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_resumes} resumes allowed. You uploaded {len(resumes)}.",
        )

    # ── 2. Extract job description text ──────────────────────────────────
    jd_text = job_description.strip()

    if not jd_text and job_description_file and job_description_file.filename:
        _validate_extension(job_description_file.filename)
        jd_bytes = await job_description_file.read()
        jd_text = extract_text_from_file(jd_bytes, job_description_file.filename)

    if not jd_text:
        raise HTTPException(
            status_code=400,
            detail="Job description is required. Provide text or upload a file.",
        )

    # ── 3. Generate session ID ───────────────────────────────────────────
    session_id = str(uuid.uuid4())
    logger.info("Session %s: Starting document ingestion", session_id)

    # ── 4. Process each resume: Extract → Chunk → Store in ChromaDB ──────
    candidate_names = []
    resume_filenames = []
    total_chunks = 0

    for resume_file in resumes:
        if not resume_file.filename:
            continue

        _validate_extension(resume_file.filename)
        file_bytes = await resume_file.read()
        _validate_size(file_bytes, resume_file.filename)

        # Extract text
        logger.info("Parsing resume: %s", resume_file.filename)
        raw_text = extract_text_from_file(file_bytes, resume_file.filename)

        if not raw_text.strip():
            logger.warning("No text extracted from %s, skipping.", resume_file.filename)
            continue

        # Extract candidate name from text (simple heuristic: first non-empty line)
        candidate_name = _extract_candidate_name(raw_text, resume_file.filename)

        # Chunk the text
        chunks = chunk_text(raw_text)
        logger.info(
            "Resume '%s' (%s): %d chunks created",
            candidate_name, resume_file.filename, len(chunks),
        )

        if not chunks:
            logger.warning("No chunks generated for %s, skipping.", resume_file.filename)
            continue

        # Store chunks in ChromaDB with metadata
        stored = add_documents(
            chunks=chunks,
            candidate_name=candidate_name,
            session_id=session_id,
        )
        total_chunks += stored

        candidate_names.append(candidate_name)
        resume_filenames.append(resume_file.filename)

    if not candidate_names:
        raise HTTPException(status_code=400, detail="No valid resume files were processed.")

    # ── 5. Store metadata in session (JD text + candidate names) ─────────
    session_store.save(session_id, jd_text, candidate_names)
    logger.info(
        "Session %s created: %d candidate(s), %d total chunks stored in ChromaDB",
        session_id, len(candidate_names), total_chunks,
    )

    return UploadResponse(
        session_id=session_id,
        message=(
            f"Successfully processed {len(candidate_names)} resume(s). "
            f"{total_chunks} chunks stored in vector database. "
            f"Use session_id to call /rank."
        ),
        job_description_loaded=True,
        resumes_loaded=len(candidate_names),
        resume_filenames=resume_filenames,
        candidate_names=candidate_names,
        chunks_stored=total_chunks,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.allowed_extensions}",
        )


def _validate_size(file_bytes: bytes, filename: str) -> None:
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File '{filename}' exceeds the {settings.max_file_size_mb} MB limit.",
        )


def _extract_candidate_name(text: str, filename: str) -> str:
    """
    Simple heuristic to extract candidate name from resume text.
    Takes the first non-empty, non-header line that looks like a name.
    Falls back to filename-based name.
    """
    lines = text.strip().split("\n")
    for line in lines[:5]:  # Check first 5 lines
        cleaned = line.strip()
        if not cleaned:
            continue
        # Skip lines that look like section headers or contact info
        if any(kw in cleaned.lower() for kw in
               ["resume", "curriculum", "cv", "phone", "email", "@", "http",
                "address", "linkedin", "github", "objective", "summary"]):
            continue
        # A name line is typically short and contains mostly letters
        if len(cleaned) <= 50 and sum(c.isalpha() or c == ' ' for c in cleaned) > len(cleaned) * 0.7:
            return cleaned.title()

    # Fallback: derive from filename
    name = Path(filename).stem
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    return name.title()
