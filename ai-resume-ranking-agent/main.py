"""
AI Multi-Resume Ranking Agent (RAG + Agent)
Main application entry point

Architecture:
  1. Document ingestion + chunking (parser.py)
  2. Vector storage (chroma_store.py - ChromaDB)
  3. Retrieval system (RAG via ChromaDB queries)
  4. Autonomous agent loop (agent.py)
  5. Ranking + explanation engine (ranker.py)
"""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, rank
from app.config import settings

# ── Configure logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# Reduce noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AI Multi-Resume Ranking Agent (RAG + Agent)",
    description=(
        "An AI-powered backend service that uses Retrieval-Augmented Generation (RAG) "
        "and an autonomous agent loop to analyze resumes against a job description. "
        "Produces ranked candidates with evidence-based scores and explainable reasoning.\n\n"
        "**Architecture:**\n"
        "- ChromaDB for vector storage of resume chunks\n"
        "- Multi-step agent reasoning loop (NOT single-pass scoring)\n"
        "- Tool-calling for document querying\n"
        "- LLM-powered requirement extraction, evidence analysis, and explanation generation"
    ),
    version="2.0.0",
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
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AI Multi-Resume Ranking Agent",
        "status": "running"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "architecture": "RAG + Agent Loop",
        "vector_store": "ChromaDB",
        "llm_provider": "Gemini",
        "llm_model": settings.gemini_model,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
