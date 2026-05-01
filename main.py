"""
AI Multi-Resume Ranking Agent
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, rank
from app.config import settings

app = FastAPI(
    title="AI Multi-Resume Ranking Agent",
    description=(
        "An AI-powered backend service that analyzes resumes against a job description "
        "and returns a ranked list of candidates with scores and explanations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(rank.router, prefix="/rank", tags=["Rank"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AI Multi-Resume Ranking Agent",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "POST /upload — Submit job description + resumes",
            "rank": "POST /rank — Rank candidates against job description",
            "docs": "GET /docs — Interactive API documentation",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
