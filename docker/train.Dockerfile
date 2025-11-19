# ============================
#  CrediMax Training Dockerfile
# ============================

FROM python:3.11-slim

# System dependencies for scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir poetry==$POETRY_VERSION

WORKDIR /app

# Copy project metadata
COPY pyproject.toml poetry.lock ./

# Copy ML project code
COPY credit_risk_mlops ./credit_risk_mlops
COPY configs ./configs
COPY data ./data

# Install dependencies
RUN poetry install --no-root --only main

# Training entrypoint
CMD ["poetry", "run", "python", "-m", "credit_risk_mlops.models.train"]
