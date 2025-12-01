#!/bin/bash
set -e

echo "🚀 Starting FastAPI..."
uvicorn credit_risk_mlops.api.server:app --host 0.0.0.0 --port 8000
