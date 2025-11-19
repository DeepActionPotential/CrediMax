# ============================
#  CrediMax API Dockerfile
# ============================

FROM python:3.11-slim

# Install system deps needed by numpy, pandas, xgboost, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir poetry==$POETRY_VERSION

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY credit_risk_mlops ./credit_risk_mlops
COPY configs ./configs
COPY artifacts ./artifacts

# Install dependencies (without dev)
RUN poetry install --no-root --only main

# Expose API port
EXPOSE 8000

# Start API server
CMD ["poetry", "run", "uvicorn", "credit_risk_mlops.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
