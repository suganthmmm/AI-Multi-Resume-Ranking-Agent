"""
AI Multi-Resume Ranking Agent
Main application entry point
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import upload, rank
from app.config import settings

BASE_DIR = Path(__file__).resolve().parent

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

# API routes
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(rank.router, prefix="/rank", tags=["Rank"])

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", tags=["UI"], include_in_schema=False)
async def serve_ui():
    """Serve the web UI."""
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
