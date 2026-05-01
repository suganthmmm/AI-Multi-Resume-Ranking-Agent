"""
Autonomous Evaluation Agent (RAG + Tool-Calling Agent Loop).

This module implements the multi-step reasoning agent that:
1. Reads the Job Description
2. Extracts key requirements (skills, experience, qualities)
3. For each requirement:
   - Generates a search query
   - Calls query_resume tool (ChromaDB retrieval)
   - Retrieves relevant chunks
   - Analyzes evidence
4. Repeats until all requirements are evaluated
5. Generates final structured output with scores and explanations

This is NOT a single-pass scorer. It performs iterative, evidence-based
evaluation through a tool-calling agent loop.
"""

import json
import re
import time
import logging
from typing import Any

from groq import Groq
from app.config import settings
from app.chroma_store import query_resume as chroma_query

logger = logging.getLogger(__name__)
_client = Groq(api_key=settings.groq_api_key)


# ── Retry helper for rate-limited API calls ──────────────────────────────
def _call_llm(system_instruction: str, contents: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    """
    Call the Groq API with automatic retry on rate-limit errors.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": contents}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "rate limit" in error_str.lower():
                # Groq has different rate limit messages, simple exponential backoff
                wait_time = 10 * (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d). Waiting %.0fs before retry...",
                    attempt + 1, max_retries, wait_time,
                )
                time.sleep(wait_time)
            else:
                raise  # Non-rate-limit error, propagate immediately
    # Final attempt without catching
    response = _client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": contents}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: REQUIREMENT EXTRACTION FROM JD
# ═══════════════════════════════════════════════════════════════════════════

REQUIREMENT_EXTRACTION_PROMPT = """You are an expert HR analyst. 
Analyze the following Job Description and extract the key evaluation criteria.

Return ONLY a valid JSON object with this schema:
{
  "role_title": "<job title>",
  "requirements": [
    {
      "category": "<skills|experience|education|projects|certifications|soft_skills|domain_knowledge>",
      "requirement": "<specific requirement>",
      "importance": "<critical|important|nice_to_have>",
      "search_query": "<single natural language query to search in resume>"
    }
  ]
}

Rules:
- Extract exactly 4-6 of the MOST important requirements (not more).
- Each requirement must have exactly 1 targeted search_query.
- Categories: skills, experience, education, projects, certifications, soft_skills, domain_knowledge.
- importance: critical (must-have), important (strongly preferred), nice_to_have (bonus).
- search_query should be a natural language phrase that would find evidence in a resume.
- Be specific: instead of "programming", say "Python programming experience".
- Focus on the requirements that best differentiate candidates.
"""


def extract_requirements(job_description: str) -> dict:
    """
    Use LLM to extract structured requirements from the job description.
    Returns a dict with 'role_title' and 'requirements' list.
    """
    logger.info("═" * 60)
    logger.info("AGENT STEP 1: Extracting requirements from Job Description")
    logger.info("═" * 60)

    try:
        raw = _call_llm(
            system_instruction=REQUIREMENT_EXTRACTION_PROMPT,
            contents=f"JOB DESCRIPTION:\n{job_description[:4000]}",
            max_tokens=1500,
            temperature=0.1,
        )
        data = _parse_json_response(raw)

        requirements = data.get("requirements", [])
        # Limit to 6 requirements max to control API usage
        if len(requirements) > 6:
            requirements = requirements[:6]
            data["requirements"] = requirements

        logger.info("Extracted %d requirements for role: %s",
                     len(requirements), data.get("role_title", "Unknown"))

        for req in requirements:
            logger.info("  → [%s] %s (importance: %s)",
                        req.get("category", "?"),
                        req.get("requirement", "?"),
                        req.get("importance", "?"))

        return data

    except Exception as exc:
        logger.error("Requirement extraction failed: %s", exc)
        return {
            "role_title": "Unknown",
            "requirements": [
                {
                    "category": "skills",
                    "requirement": "General technical skills",
                    "importance": "important",
                    "search_query": "technical skills programming",
                },
                {
                    "category": "experience",
                    "requirement": "Work experience",
                    "importance": "important",
                    "search_query": "work experience professional",
                },
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: AGENT TOOL — QUERY RESUME (RAG RETRIEVAL)
# ═══════════════════════════════════════════════════════════════════════════

def tool_query_resume(query: str, candidate_name: str, session_id: str) -> dict:
    """
    Agent tool: Search ChromaDB for relevant resume chunks.

    This is the tool the agent calls during its reasoning loop.
    It retrieves the top relevant chunks from a candidate's resume
    stored in the vector database.

    Args:
        query: Natural language search query.
        candidate_name: Which candidate's resume to search.
        session_id: Session scope.

    Returns:
        Dict with 'query', 'candidate', 'results' list, and 'num_results'.
    """
    logger.info("  🔍 TOOL CALL: query_resume('%s', candidate='%s')", query, candidate_name)

    results = chroma_query(
        query=query,
        session_id=session_id,
        candidate_name=candidate_name,
        n_results=3,
    )

    logger.info("  📄 Retrieved %d chunks", len(results))
    for i, r in enumerate(results):
        snippet = r["text"][:120].replace("\n", " ")
        logger.info("    Chunk %d (score=%.3f): %s...", i + 1, r["score"], snippet)

    return {
        "query": query,
        "candidate": candidate_name,
        "results": results,
        "num_results": len(results),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 & 4 & 5: BATCHED EVIDENCE ANALYSIS & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

ALL_CANDIDATES_EVALUATION_PROMPT = """You are an expert HR evaluator performing evidence-based resume analysis.

You will be given a list of Job Requirements and the relevant resume chunks retrieved for MULTIPLE candidates.
Analyze each candidate's fit based ONLY on the provided chunks.

IMPORTANT: Each candidate MUST receive a DIFFERENT score reflecting their actual fit. Do NOT give the same score to all candidates.

Return ONLY a valid JSON object matching this schema:
{
  "candidates": [
    {
      "name": "<candidate name>",
      "requirement_evaluations": [
        {
          "requirement": "<the requirement being evaluated>",
          "met": <true|false|"partial">,
          "score": <0 to 100>,
          "evidence": ["<specific evidence quote or finding>"],
          "reasoning": "<1-2 sentences explaining assessment>"
        }
      ],
      "final_score": <0 to 100 overall score based on individual scores and requirement importance>,
      "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
      "weaknesses": ["<weakness 1>", "<weakness 2>"],
      "explanation": "<3-4 sentence paragraph explaining the candidate's overall fit and key differentiators.>"
    }
  ]
}

Rules:
- Base your assessment ONLY on the provided resume chunks.
- If no relevant evidence is found for a requirement, set met=false, score=0-20.
- Provide specific quotes or references as evidence.
- strengths: 2-5 specific, evidence-based bullet points.
- weaknesses: 1-3 genuine gaps.
- explanation: specific to EACH candidate, must be different.
- Scores MUST vary between candidates based on evidence quality.
- A candidate with more relevant evidence should score higher.
"""


def evaluate_all_candidates(
    candidate_names: list[str],
    session_id: str,
    requirements: list[dict],
    job_description: str,
) -> list[dict]:
    """
    Evaluate ALL candidates in a single LLM call to minimize API usage.
    Gathers evidence for all candidates first, then sends one evaluation request.
    """
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  EVALUATING ALL %d CANDIDATES                            ║", len(candidate_names))
    logger.info("╚" + "═" * 58 + "╝")

    # ── Gather evidence for all candidates ─────────────────────────────
    all_candidate_evidence = {}

    for candidate_name in candidate_names:
        logger.info("── Gathering evidence for: %s ──", candidate_name)
        req_chunks_map = {}

        for requirement in requirements:
            req_desc = requirement.get("requirement", "Unknown requirement")
            # Use single search_query (not list) to reduce embedding API calls
            query = requirement.get("search_query", req_desc)
            if isinstance(query, list):
                query = query[0] if query else req_desc

            tool_result = tool_query_resume(query, candidate_name, session_id)
            req_chunks_map[req_desc] = tool_result["results"]
            logger.info("   Retrieved %d chunks for: %s", len(tool_result["results"]), req_desc)

        all_candidate_evidence[candidate_name] = req_chunks_map

    # ── Build single prompt for ALL candidates ─────────────────────────
    logger.info("🧠 Batch evaluating ALL %d candidates in single LLM call...", len(candidate_names))

    prompt_lines = [
        f"JOB DESCRIPTION EXCERPT:\n{job_description[:800]}\n",
        "=" * 60,
    ]

    for candidate_name in candidate_names:
        prompt_lines.append(f"\n{'━' * 50}")
        prompt_lines.append(f"CANDIDATE: {candidate_name}")
        prompt_lines.append(f"{'━' * 50}")

        req_chunks_map = all_candidate_evidence[candidate_name]

        for req in requirements:
            req_desc = req.get("requirement", "Unknown")
            category = req.get("category", "general")
            importance = req.get("importance", "important")
            chunks = req_chunks_map.get(req_desc, [])

            prompt_lines.append(f"\n--- REQUIREMENT: {req_desc} (Category: {category}, Importance: {importance}) ---")
            if not chunks:
                prompt_lines.append("(No relevant resume content found for this requirement)")
            else:
                for i, c in enumerate(chunks[:2]):  # limit to top 2 chunks to save tokens
                    prompt_lines.append(f"[Chunk {i+1}, relevance={c['score']:.3f}]: {c['text'][:300]}")

    user_prompt = "\n".join(prompt_lines)

    try:
        raw = _call_llm(
            system_instruction=ALL_CANDIDATES_EVALUATION_PROMPT,
            contents=user_prompt,
            max_tokens=3000,
            temperature=0.2,
        )
        result = _parse_json_response(raw)

        candidates_data = result.get("candidates", [])

        # Build results list
        results = []
        for cand_data in candidates_data:
            name = cand_data.get("name", "Unknown")
            final_score = float(cand_data.get("final_score", 0.0))
            final_score = round(min(100, max(0, final_score)), 1)

            all_evidence = []
            for ev in cand_data.get("requirement_evaluations", []):
                if ev.get("evidence"):
                    all_evidence.extend(ev["evidence"])

            logger.info("── CANDIDATE SCORE: %s → %.1f/100 ──", name, final_score)

            results.append({
                "name": name,
                "score": final_score,
                "strengths": cand_data.get("strengths", []),
                "weaknesses": cand_data.get("weaknesses", []),
                "evidence": all_evidence[:10],
                "explanation": cand_data.get("explanation", ""),
                "requirement_evaluations": cand_data.get("requirement_evaluations", []),
            })

        # If some candidates are missing from the LLM response, add them with fallback
        responded_names = {r["name"] for r in results}
        for name in candidate_names:
            if name not in responded_names:
                logger.warning("Candidate '%s' missing from LLM response, adding with fallback score", name)
                results.append({
                    "name": name,
                    "score": 30.0,
                    "strengths": ["Profile reviewed"],
                    "weaknesses": ["Evaluation data limited"],
                    "evidence": [],
                    "explanation": f"Limited evaluation data for {name}.",
                    "requirement_evaluations": [],
                })

        return results

    except Exception as exc:
        logger.error("⚠️ Batch evaluation failed: %s", exc)
        # Return fallback for all candidates
        return [
            {
                "name": name,
                "score": 10.0,
                "strengths": ["Review initiated"],
                "weaknesses": ["Evaluation incomplete due to an error"],
                "evidence": [],
                "explanation": f"Failed to fully process {name}.",
                "requirement_evaluations": [],
            }
            for name in candidate_names
        ]


# ── Legacy single-candidate function (kept for backward compatibility) ───

def evaluate_candidate(
    candidate_name: str,
    session_id: str,
    requirements: list[dict],
    job_description: str,
) -> dict:
    """
    Run the full agent evaluation loop for a single candidate.
    """
    results = evaluate_all_candidates(
        candidate_names=[candidate_name],
        session_id=session_id,
        requirements=requirements,
        job_description=job_description,
    )
    return results[0] if results else {
        "name": candidate_name,
        "score": 10.0,
        "strengths": ["Review initiated"],
        "weaknesses": ["Evaluation incomplete"],
        "evidence": [],
        "explanation": f"Failed to process {candidate_name}.",
    }
