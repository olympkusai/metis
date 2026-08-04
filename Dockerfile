FROM python:3.14-slim

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

# Expose port
EXPOSE 8082

# Run the application
CMD ["uvicorn", "metis.main:app", "--host", "0.0.0.0", "--port", "8082", "--loop", "asyncio"]
