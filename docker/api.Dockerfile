FROM python:3.11-slim

# ----------------------------------------------------
# Install system dependencies
# ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# Install Poetry
# ----------------------------------------------------
ENV POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

# ----------------------------------------------------
# Set working dir
# ----------------------------------------------------
WORKDIR /app

# ----------------------------------------------------
# Copy project files
# ----------------------------------------------------
COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

# Now copy the full project
COPY . /app

# Create required folders
RUN mkdir -p /app/artifacts /app/mlruns

# Expose API port
EXPOSE 8000

# Start script
COPY scripts/start_api.sh /app/scripts/start_api.sh
RUN chmod +x /app/scripts/start_api.sh

CMD ["/app/scripts/start_api.sh"]
