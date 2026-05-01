"""
ChromaDB Vector Store for Resume Chunks.

Provides persistent vector storage for resume document chunks,
with metadata filtering by session_id and candidate_name.
Uses ChromaDB's built-in default embedding function.
"""

import logging
from pathlib import Path

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from app.config import settings

logger = logging.getLogger(__name__)

# Persistent storage directory
_CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_data"
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Singleton ChromaDB client (persistent)
_client = chromadb.PersistentClient(path=str(_CHROMA_DIR))

# Collection name
COLLECTION_NAME = "resume_chunks"


def _get_collection():
    """Get or create the resume chunks collection."""
    # Using Chroma's default sentence-transformers embedding function
    # which runs locally and requires no API key.
    ef = embedding_functions.DefaultEmbeddingFunction()
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )


def add_documents(
    chunks: list[str],
    candidate_name: str,
    session_id: str,
) -> int:
    """
    Store text chunks in ChromaDB with metadata.

    Args:
        chunks: List of text chunks to store.
        candidate_name: Name of the candidate these chunks belong to.
        session_id: Session identifier for isolation.

    Returns:
        Number of chunks stored.
    """
    if not chunks:
        logger.warning("No chunks to store for %s", candidate_name)
        return 0

    collection = _get_collection()

    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{session_id}_{candidate_name}_{i}"
        ids.append(chunk_id)
        metadatas.append({
            "candidate_name": candidate_name,
            "session_id": session_id,
            "chunk_index": i,
        })

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas,
    )

    logger.info(
        "Stored %d chunks for candidate '%s' in session '%s'",
        len(chunks), candidate_name, session_id,
    )
    return len(chunks)


def query_resume(
    query: str,
    session_id: str,
    candidate_name: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Query ChromaDB for relevant resume chunks.

    Args:
        query: The search query (e.g., "Python experience").
        session_id: Session to search within.
        candidate_name: Candidate whose resume to search.
        n_results: Number of top results to return.

    Returns:
        List of dicts with keys: 'text', 'score', 'metadata'.
    """
    collection = _get_collection()

    # Count available documents for this candidate in this session
    try:
        existing = collection.get(
            where={
                "$and": [
                    {"session_id": {"$eq": session_id}},
                    {"candidate_name": {"$eq": candidate_name}},
                ]
            },
        )
        available_count = len(existing["ids"]) if existing and existing["ids"] else 0
    except Exception:
        available_count = n_results  # fallback

    if available_count == 0:
        logger.warning(
            "No documents found for candidate '%s' in session '%s'",
            candidate_name, session_id,
        )
        return []

    # Adjust n_results to not exceed available documents
    actual_n = min(n_results, available_count)

    results = collection.query(
        query_texts=[query],
        n_results=actual_n,
        where={
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"candidate_name": {"$eq": candidate_name}},
            ]
        },
    )

    output = []
    if results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)

        for doc, dist, meta in zip(docs, distances, metas):
            # ChromaDB cosine distance: lower = more similar
            # Convert to similarity score (0-1)
            similarity = max(0.0, 1.0 - dist)
            output.append({
                "text": doc,
                "score": round(similarity, 4),
                "metadata": meta,
            })

    logger.info(
        "Query '%s' for '%s': returned %d results",
        query[:50], candidate_name, len(output),
    )
    return output


def delete_session(session_id: str) -> int:
    """
    Delete all chunks belonging to a session.

    Returns:
        Number of deleted chunks (approximate).
    """
    collection = _get_collection()
    try:
        existing = collection.get(
            where={"session_id": {"$eq": session_id}},
        )
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info("Deleted %d chunks for session '%s'", len(existing["ids"]), session_id)
            return len(existing["ids"])
    except Exception as exc:
        logger.warning("Failed to delete session '%s': %s", session_id, exc)
    return 0


def get_candidates_in_session(session_id: str) -> list[str]:
    """Return a list of unique candidate names stored for a session."""
    collection = _get_collection()
    try:
        existing = collection.get(
            where={"session_id": {"$eq": session_id}},
        )
        if existing and existing["metadatas"]:
            names = set(m.get("candidate_name", "") for m in existing["metadatas"])
            return sorted(names - {""})
    except Exception as exc:
        logger.warning("Failed to list candidates for session '%s': %s", session_id, exc)
    return []
