import os
import json
import time
import requests
from docx import Document

# ── 1. Create Mock Documents ─────────────────────────────────────────────────

def create_mock_docx(filename: str, text: str):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filename)
    print(f"Created {filename}")

print("--- Generating Mock Data ---")

# Job Description
jd_text = """
Job Title: Senior Python Backend Developer
Requirements:
- 5+ years of experience in backend development
- Strong expertise in Python and FastAPI
- Experience with vector databases like ChromaDB or Pinecone
- Knowledge of Langchain and building RAG pipelines
- Good understanding of Docker and deployment
"""

# Resume 1: Perfect Match
resume_1 = """
Alice Smith
Senior Backend Engineer
Experience:
- 6 years of backend engineering using Python.
- Built scalable APIs using FastAPI and deployed them with Docker.
- Integrated AI workflows using Langchain and OpenAI.
- Designed vector search functionality using ChromaDB.
Education:
- MS in Computer Science
"""
create_mock_docx("resume_alice.docx", resume_1)

# Resume 2: Partial Match
resume_2 = """
Bob Jones
Software Developer
Experience:
- 3 years of software development in Java and Spring Boot.
- Familiar with Python scripting.
- Built simple REST APIs. No experience with LLMs or vector databases.
Education:
- BS in Computer Science
"""
create_mock_docx("resume_bob.docx", resume_2)

# Resume 3: Good Match with some gaps
resume_3 = """
Charlie Brown
Machine Learning Engineer
Experience:
- 4 years of Python development.
- Built and fine-tuned LLMs. Used Pinecone for vector embeddings.
- Very familiar with Langchain.
- Has not used FastAPI directly, usually uses Flask.
Education:
- PhD in Data Science
"""
create_mock_docx("resume_charlie.docx", resume_3)


# ── 2. Test the API Pipeline ─────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"

print("\n--- Testing /upload endpoint ---")
upload_url = f"{BASE_URL}/upload"

files = [
    ("resumes", ("resume_alice.docx", open("resume_alice.docx", "rb"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    ("resumes", ("resume_bob.docx", open("resume_bob.docx", "rb"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    ("resumes", ("resume_charlie.docx", open("resume_charlie.docx", "rb"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
]

data = {
    "job_description": jd_text
}

response = requests.post(upload_url, data=data, files=files)

if response.status_code != 200:
    print(f"Upload failed: {response.status_code}")
    print(response.text)
    exit(1)

upload_data = response.json()
print("Upload Successful!")
print(f"Session ID: {upload_data['session_id']}")
print(f"Chunks stored in ChromaDB: {upload_data['chunks_stored']}")
print(f"Candidates found: {upload_data['candidate_names']}")

session_id = upload_data["session_id"]


print("\n--- Testing /rank endpoint ---")
print("This will trigger the Agent Loop. It might take 30-60 seconds depending on the LLM API...")
rank_url = f"{BASE_URL}/rank"

start_time = time.time()
rank_response = requests.post(rank_url, json={"session_id": session_id})
end_time = time.time()

if rank_response.status_code != 200:
    print(f"Ranking failed: {rank_response.status_code}")
    print(rank_response.text)
    exit(1)

rank_data = rank_response.json()
print(f"\nRanking Completed in {end_time - start_time:.1f} seconds!\n")

print("="*60)
print(f"RANKING SUMMARY: {rank_data['ranking_summary']}")
print("="*60)

for i, candidate in enumerate(rank_data["candidates"]):
    print(f"\n[{i+1}] {candidate['name'].upper()} - Score: {candidate['score']}/100")
    print(f"    Explanation: {candidate['explanation']}")
    
    print("    Strengths:")
    for strength in candidate['strengths']:
        print(f"      + {strength}")
        
    print("    Weaknesses:")
    for weakness in candidate['weaknesses']:
        print(f"      - {weakness}")
        
    print("    Top Evidence Found:")
    for ev in candidate['evidence'][:3]:
        print(f"      > {ev}")

# Cleanup files
for file in ["resume_alice.docx", "resume_bob.docx", "resume_charlie.docx"]:
    if os.path.exists(file):
        os.remove(file)
