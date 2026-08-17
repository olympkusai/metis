FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for C extensions
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY ./metis ./metis

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose port (Railway injects PORT env var; default 8082 for local/docker-compose)
ENV PORT=8082
EXPOSE 8082

# Run the application — port from env (Railway injects PORT)
CMD ["sh", "-c", "uvicorn metis.main:app --host 0.0.0.0 --port ${PORT:-8082} --loop asyncio"]
