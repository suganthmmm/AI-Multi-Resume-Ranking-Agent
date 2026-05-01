"""
POST /upload
Accepts a job description (text or file) plus up to 5 resume files.
Extracts features and stores them in the session store.
"""

import uuid
import logging
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.models import UploadResponse
from app.parser import extract_text_from_file
from app.extractor import extract_resume_features, extract_jd_features
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
    **Step 1 of 2 — Upload**

    - Supply the job description as plain text **or** a file (PDF/DOCX).
    - Upload 1–5 resume files (PDF/DOCX).
    - Returns a `session_id` to use in the subsequent `/rank` call.
    """

    # ── 1. Validate resumes count ────────────────────────────────────────────
    if not resumes or (len(resumes) == 1 and not resumes[0].filename):
        raise HTTPException(status_code=400, detail="At least one resume file is required.")
    if len(resumes) > settings.max_resumes:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_resumes} resumes allowed. You uploaded {len(resumes)}.",
        )

    # ── 2. Extract job description text ─────────────────────────────────────
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

    # ── 3. Parse job description with AI ────────────────────────────────────
    logger.info("Extracting JD features…")
    extracted_jd = extract_jd_features(jd_text)

    # ── 4. Parse each resume ─────────────────────────────────────────────────
    extracted_resumes = []
    resume_filenames = []

    for resume_file in resumes:
        if not resume_file.filename:
            continue
        _validate_extension(resume_file.filename)
        _validate_size(await resume_file.read(), resume_file.filename)
        # Re-read after validation (already consumed above via read())
        await resume_file.seek(0)
        file_bytes = await resume_file.read()

        logger.info("Parsing resume: %s", resume_file.filename)
        raw_text = extract_text_from_file(file_bytes, resume_file.filename)
        parsed = extract_resume_features(raw_text, resume_file.filename)
        extracted_resumes.append(parsed)
        resume_filenames.append(resume_file.filename)

    if not extracted_resumes:
        raise HTTPException(status_code=400, detail="No valid resume files were processed.")

    # ── 5. Store in session ──────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    session_store.save(session_id, extracted_jd, extracted_resumes)
    logger.info("Session %s created with %d resume(s).", session_id, len(extracted_resumes))

    return UploadResponse(
        session_id=session_id,
        message=f"Successfully processed {len(extracted_resumes)} resume(s). Use session_id to call /rank.",
        job_description_loaded=True,
        resumes_loaded=len(extracted_resumes),
        resume_filenames=resume_filenames,
        extracted_job_role=extracted_jd.role_title,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _validate_extension(filename: str) -> None:
    from pathlib import Path
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
