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
import logging
from typing import Any

import google.generativeai as genai
from app.config import settings
from app.chroma_store import query_resume as chroma_query

logger = logging.getLogger(__name__)
genai.configure(api_key=settings.gemini_api_key)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: REQUIREMENT EXTRACTION FROM JD
# ═══════════════════════════════════════════════════════════════════════════

REQUIREMENT_EXTRACTION_PROMPT = """You are an expert HR analyst. 
Analyze the following Job Description and extract ALL key evaluation criteria.

Return ONLY a valid JSON object with this schema:
{
  "role_title": "<job title>",
  "requirements": [
    {
      "category": "<skills|experience|education|projects|certifications|soft_skills|domain_knowledge>",
      "requirement": "<specific requirement>",
      "importance": "<critical|important|nice_to_have>",
      "search_queries": ["<query1 to search in resume>", "<query2>"]
    }
  ]
}

Rules:
- Extract 8-15 specific, searchable requirements.
- Each requirement must have 1-2 targeted search queries.
- Categories: skills, experience, education, projects, certifications, soft_skills, domain_knowledge.
- importance: critical (must-have), important (strongly preferred), nice_to_have (bonus).
- search_queries should be natural language phrases that would find evidence in a resume.
- Be specific: instead of "programming", say "Python programming experience".
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
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=REQUIREMENT_EXTRACTION_PROMPT,
            generation_config=genai.GenerationConfig(
                max_output_tokens=2000,
                temperature=0.1,
            )
        )
        response = model.generate_content(f"JOB DESCRIPTION:\n{job_description[:4000]}")
        raw = response.text.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        data = json.loads(raw)

        requirements = data.get("requirements", [])
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
                    "search_queries": ["technical skills", "programming experience"],
                },
                {
                    "category": "experience",
                    "requirement": "Work experience",
                    "importance": "important",
                    "search_queries": ["work experience", "professional experience"],
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
        n_results=5,
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
# STEP 3: EVIDENCE ANALYSIS (PER REQUIREMENT)
# ═══════════════════════════════════════════════════════════════════════════

EVIDENCE_ANALYSIS_PROMPT = """You are an expert HR evaluator performing evidence-based resume analysis.

Given a specific job requirement and retrieved resume chunks, analyze whether the candidate meets this requirement.

Return ONLY a valid JSON object:
{
  "requirement": "<the requirement being evaluated>",
  "met": <true|false|"partial">,
  "confidence": <0.0 to 1.0>,
  "score": <0 to 100>,
  "evidence": ["<specific evidence quote or finding from the resume chunks>"],
  "reasoning": "<2-3 sentences explaining your assessment>"
}

Rules:
- Base your assessment ONLY on the provided resume chunks (evidence).
- If no relevant evidence is found, set met=false, confidence=low, score=0-20.
- Provide specific quotes or references as evidence.
- Be objective and fair.
"""


def analyze_evidence(
    requirement: dict,
    retrieved_chunks: list[dict],
    candidate_name: str,
) -> dict:
    """
    Use LLM to analyze whether retrieved chunks satisfy a requirement.
    This is part of the agent's reasoning loop.
    """
    chunks_text = "\n---\n".join(
        f"[Chunk {i+1}, relevance={c['score']:.3f}]:\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )

    if not chunks_text.strip():
        chunks_text = "(No relevant resume content found for this query)"

    user_prompt = f"""
CANDIDATE: {candidate_name}

REQUIREMENT:
- Category: {requirement.get('category', 'general')}
- Description: {requirement.get('requirement', 'N/A')}
- Importance: {requirement.get('importance', 'important')}

RETRIEVED RESUME CHUNKS:
{chunks_text}

Analyze the evidence and provide your assessment.
"""

    try:
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=EVIDENCE_ANALYSIS_PROMPT,
            generation_config=genai.GenerationConfig(
                max_output_tokens=500,
                temperature=0.2,
            )
        )
        response = model.generate_content(user_prompt)
        raw = response.text.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        result = json.loads(raw)

        logger.info("    ✅ Evidence analysis: met=%s, score=%s, confidence=%.2f",
                     result.get("met"), result.get("score", 0), result.get("confidence", 0))

        return result

    except Exception as exc:
        logger.warning("    ⚠️ Evidence analysis failed: %s", exc)
        return {
            "requirement": requirement.get("requirement", "Unknown"),
            "met": False,
            "confidence": 0.2,
            "score": 10,
            "evidence": [],
            "reasoning": "Analysis could not be completed due to an error.",
        }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: AGENT LOOP — MULTI-STEP REASONING FOR ONE CANDIDATE
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_candidate(
    candidate_name: str,
    session_id: str,
    requirements: list[dict],
    job_description: str,
) -> dict:
    """
    Run the full agent evaluation loop for a single candidate.

    This is the core agent loop that:
    1. Iterates over each extracted requirement
    2. For each requirement, generates search queries
    3. Calls the query_resume tool (RAG retrieval from ChromaDB)
    4. Analyzes the retrieved evidence
    5. Accumulates scores and evidence
    6. Generates a final assessment

    Returns:
        Dict with 'name', 'score', 'strengths', 'weaknesses',
        'evidence', 'explanation', 'requirement_evaluations'.
    """
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  EVALUATING CANDIDATE: %-34s ║", candidate_name)
    logger.info("╚" + "═" * 58 + "╝")

    evaluations = []
    all_evidence = []
    total_weighted_score = 0.0
    total_weight = 0.0

    # Importance weights
    importance_weights = {
        "critical": 3.0,
        "important": 2.0,
        "nice_to_have": 1.0,
    }

    for i, requirement in enumerate(requirements, start=1):
        req_desc = requirement.get("requirement", "Unknown requirement")
        category = requirement.get("category", "general")
        importance = requirement.get("importance", "important")
        search_queries = requirement.get("search_queries", [req_desc])

        logger.info("")
        logger.info("── Requirement %d/%d: %s ──", i, len(requirements), req_desc)
        logger.info("   Category: %s | Importance: %s", category, importance)

        # ── AGENT STEP: Generate & execute search queries ────────────────
        all_chunks = []
        for query in search_queries:
            logger.info("   Querying resume: '%s'", query)
            tool_result = tool_query_resume(query, candidate_name, session_id)
            all_chunks.extend(tool_result["results"])

        # De-duplicate chunks by text content
        seen_texts = set()
        unique_chunks = []
        for chunk in all_chunks:
            text_key = chunk["text"][:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_chunks.append(chunk)

        logger.info("   Retrieved %d unique chunks for this requirement", len(unique_chunks))

        # ── AGENT STEP: Analyze evidence ─────────────────────────────────
        logger.info("   Analyzing evidence for: %s", req_desc)
        evaluation = analyze_evidence(requirement, unique_chunks, candidate_name)
        evaluations.append(evaluation)

        # Accumulate weighted score
        weight = importance_weights.get(importance, 2.0)
        score = float(evaluation.get("score", 0))
        total_weighted_score += score * weight
        total_weight += weight

        # Collect evidence
        evidence_list = evaluation.get("evidence", [])
        if evidence_list:
            for ev in evidence_list:
                all_evidence.append(f"[{category}] {ev}")

    # ── AGENT STEP: Calculate final score ────────────────────────────────
    final_score = (total_weighted_score / total_weight) if total_weight > 0 else 0.0
    final_score = round(min(100, max(0, final_score)), 1)

    logger.info("")
    logger.info("── CANDIDATE SCORE: %s → %.1f/100 ──", candidate_name, final_score)

    # ── AGENT STEP: Generate final assessment ────────────────────────────
    assessment = generate_final_assessment(
        candidate_name=candidate_name,
        evaluations=evaluations,
        final_score=final_score,
        job_description=job_description,
    )

    return {
        "name": candidate_name,
        "score": final_score,
        "strengths": assessment.get("strengths", []),
        "weaknesses": assessment.get("weaknesses", []),
        "evidence": all_evidence[:10],  # Cap at 10 evidence items
        "explanation": assessment.get("explanation", ""),
        "requirement_evaluations": evaluations,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: FINAL ASSESSMENT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

FINAL_ASSESSMENT_PROMPT = """You are a senior HR analyst writing a candidate evaluation summary.

Based on the detailed requirement-by-requirement evaluation results below,
produce a concise overall assessment.

Return ONLY a valid JSON object:
{
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "explanation": "<3-4 sentence paragraph explaining the candidate's overall fit, key differentiators, and main gaps. Reference specific skills or evidence.>"
}

Rules:
- strengths: 2-5 specific, evidence-based bullet points.
- weaknesses: 1-3 genuine gaps (be constructive).
- explanation: must be specific to THIS candidate, not generic.
- Reference actual findings from the evaluations.
"""


def generate_final_assessment(
    candidate_name: str,
    evaluations: list[dict],
    final_score: float,
    job_description: str,
) -> dict:
    """
    Generate the final strengths/weaknesses/explanation for a candidate
    based on all the requirement evaluations.
    """
    logger.info("Generating final assessment for %s...", candidate_name)

    evals_text = ""
    for i, ev in enumerate(evaluations, 1):
        evals_text += (
            f"\n{i}. {ev.get('requirement', 'N/A')}\n"
            f"   Met: {ev.get('met', 'unknown')} | Score: {ev.get('score', 0)}/100\n"
            f"   Evidence: {', '.join(ev.get('evidence', ['None']))}\n"
            f"   Reasoning: {ev.get('reasoning', 'N/A')}\n"
        )

    user_prompt = f"""
CANDIDATE: {candidate_name}
OVERALL SCORE: {final_score}/100

REQUIREMENT-BY-REQUIREMENT EVALUATIONS:
{evals_text}

JOB DESCRIPTION EXCERPT:
{job_description[:1500]}

Generate the final assessment JSON.
"""

    try:
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=FINAL_ASSESSMENT_PROMPT,
            generation_config=genai.GenerationConfig(
                max_output_tokens=600,
                temperature=0.3,
            )
        )
        response = model.generate_content(user_prompt)
        raw = response.text.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        result = json.loads(raw)
        return result

    except Exception as exc:
        logger.warning("Final assessment generation failed for %s: %s", candidate_name, exc)
        return {
            "strengths": ["Profile reviewed"],
            "weaknesses": ["Assessment could not be fully generated"],
            "explanation": (
                f"{candidate_name} received a score of {final_score}/100. "
                "Detailed assessment generation encountered an error."
            ),
        }
