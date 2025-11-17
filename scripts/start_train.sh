#!/bin/bash
set -e

echo "🚀 Training model..."
poetry run python -m credit_risk_mlops.models.train
