"""Central settings for the backend, resolved once from environment variables — matches
the device-agnostic pattern in src/common/config.py: nothing in the app code hardcodes a
path or host, everything reads from here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# backend/app/core/config.py -> repo root is 3 levels up
REPO_ROOT = Path(__file__).resolve().parents[3]

# Loads backend/.env if present (see backend/.env.example); a no-op when it doesn't exist,
# and never overrides a variable already set in the real environment (e.g. by docker-compose
# or the OS), so deployment env vars always win over a stray .env file.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


class Settings:
    content_dir: Path = Path(os.environ.get("MOFSEMNET_CONTENT_DIR", REPO_ROOT / "content"))
    data_dir: Path = Path(os.environ.get("MOFSEMNET_DATA_DIR", REPO_ROOT / "data"))
    outputs_dir: Path = Path(os.environ.get("MOFSEMNET_OUTPUTS_DIR", REPO_ROOT / "outputs"))
    papers_dir: Path = Path(os.environ.get("MOFSEMNET_PAPERS_DIR", REPO_ROOT / "papers"))

    # Comma-separated list; defaults cover the Vite dev server and a same-origin production
    # deploy (where the frontend is served by this same FastAPI app, so "*" origin issues
    # don't arise, but a dev split across ports needs explicit CORS).
    cors_origins: list[str] = os.environ.get(
        "MOFSEMNET_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")

    # Which language model backs the live embedding demo on /implemented. Must be one of
    # src.common.config.LANGUAGE_MODEL_CHOICES.
    demo_language_model: str = os.environ.get("MOFSEMNET_DEMO_MODEL", "chemberta")

    # How many nearest real MOFs the live demo returns for a submitted SMILES string.
    demo_top_k: int = int(os.environ.get("MOFSEMNET_DEMO_TOP_K", "5"))


settings = Settings()
