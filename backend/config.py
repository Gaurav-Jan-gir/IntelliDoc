"""Configuration module for IntelliDoc.

Loads environment variables and application settings using Pydantic Settings
with seamless fallback for uninstalled environments.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Application configuration settings using Pydantic BaseSettings."""

        GOOGLE_API_KEY: Optional[str] = ""
        EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
        CHUNK_SIZE: int = 500
        CHUNK_OVERLAP: int = 50
        TOP_K: int = 3

        HOST: str = "127.0.0.1"
        PORT: int = 8000
        STREAMLIT_PORT: int = 8501
        API_BASE_URL: str = "http://127.0.0.1:8000"

        BASE_DIR: Path = Path(__file__).resolve().parent.parent
        DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
        VECTORSTORE_DIR: Path = Path(__file__).resolve().parent.parent / "vectorstore"
        DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "intellidoc.db"

        model_config = SettingsConfigDict(
            env_file=Path(__file__).resolve().parent.parent / ".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        def init_directories(self) -> None:
            """Create necessary storage directories if they do not exist."""
            self.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
            if self.DB_PATH.parent:
                self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

except ImportError:
    # Graceful fallback when pydantic-settings is not yet installed in current environment
    class Settings:  # type: ignore
        """Pure-Python fallback settings."""

        def __init__(self) -> None:
            self._load_env()
            self.GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY", "")
            self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
            self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
            self.TOP_K: int = int(os.getenv("TOP_K", "3"))

            self.HOST: str = os.getenv("HOST", "127.0.0.1")
            self.PORT: int = int(os.getenv("PORT", "8000"))
            self.STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
            self.API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

            self.BASE_DIR: Path = Path(__file__).resolve().parent.parent
            self.DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
            self.VECTORSTORE_DIR: Path = Path(__file__).resolve().parent.parent / "vectorstore"
            self.DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "intellidoc.db"

        def _load_env(self) -> None:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip())

        def init_directories(self) -> None:
            self.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
            if self.DB_PATH.parent:
                self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.init_directories()

