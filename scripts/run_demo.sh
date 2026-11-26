#!/usr/bin/env bash
# Start the MOFSemNet web app (landing page, Live Demo, Research Explorer) on one URL.
#   bash scripts/run_demo.sh [port]        default port 8010, then open http://localhost:8010
# Unpacks mofsemnet_models.zip into outputs/models/ if needed and builds the frontend if it has not been built.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8010}"
PY="${PYTHON:-.venv/bin/python}"
MODELS=outputs/models/mofsemnet_models
if [ ! -f "$MODELS/manifest.json" ]; then
  if [ -f mofsemnet_models.zip ]; then
    echo "Unpacking mofsemnet_models.zip into outputs/models/ ..."
    mkdir -p outputs/models && unzip -q -o mofsemnet_models.zip -d outputs/models
  fi
  [ -f "$MODELS/manifest.json" ] || echo "WARNING: no saved models in $MODELS: the Live Demo will not predict (run MOFSemNet_server_save_models.ipynb, then unzip its output there)."
fi
if [ ! -f frontend/dist/index.html ] || [ frontend/public/mofworld.html -nt frontend/dist/mofworld.html ]; then
  echo "Building the frontend ..."
  (cd frontend && { [ -d node_modules ] || npm install; } && npm run build)
fi
echo "MOFSemNet on http://localhost:$PORT (models load in the background, about 10 s)"
PYTHONPATH=. exec "$PY" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port "$PORT"
