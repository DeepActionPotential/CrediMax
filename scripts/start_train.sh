#!/bin/bash
set -e

echo "🚀 Training model..."
poetry run python -m src.models.train
