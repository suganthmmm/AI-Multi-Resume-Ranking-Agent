"""
Configuration settings for the AI Resume Ranking Agent.
Reads from environment variables or .env file.
"""

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Upload constraints
    max_resumes: int = 5
    max_file_size_mb: int = 10
    upload_dir: str = "uploads"
    allowed_extensions: list[str] = [".pdf", ".docx", ".doc"]

    # Scoring weights (must sum to 1.0)
    weight_skills: float = 0.4
    weight_experience: float = 0.3
    weight_projects: float = 0.2
    weight_education: float = 0.1

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.upload_dir, exist_ok=True)
