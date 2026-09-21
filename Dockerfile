# Reproducible environment for MOFSemNet — a university-server / grader-friendly baseline,
# not a deployment image. CPU-only by default (see requirements.txt); a GPU machine can still
# use this image's dependency set as a base and swap the torch wheel for a CUDA build (see
# README.md's device configuration notes) rather than needing a separate Dockerfile.
FROM python:3.11-slim

# RDKit and matplotlib both need a small set of system libraries beyond the Python wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . .

# Editable install so `python -m pytest` and `python scripts/run_step*.py` work the same way
# inside the container as in a local checkout, without needing PYTHONPATH tricks.
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "pytest", "-q"]
