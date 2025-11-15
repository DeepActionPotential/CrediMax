#!/bin/bash
set -e

echo "🚀 Starting FastAPI..."
poetry run uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
