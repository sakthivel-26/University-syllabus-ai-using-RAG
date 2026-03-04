"""Configuration management for the Syllabus AI Assistant."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:

    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
    )

    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")

    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "z-ai/glm-4.5-air:free"
    )

    CHROMA_DB_PATH: str = os.getenv(
        "CHROMA_DB_PATH",
        "./chroma_db"
    )

    COLLECTION_NAME: str = os.getenv(
        "COLLECTION_NAME",
        "syllabus_documents"
    )

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "6"))

    @classmethod
    def validate(cls):

        if cls.LLM_PROVIDER != "openrouter":
            raise ValueError("Only OpenRouter provider supported")

        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY missing in .env"
            )

        Path(cls.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_chroma_db_path(cls) -> Path:
        return Path(cls.CHROMA_DB_PATH)