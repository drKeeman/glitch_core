# Multi-stage build for development and production
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install uv

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --frozen

# Development stage
FROM base AS development

# Install development dependencies
RUN uv sync --frozen --group dev

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY experiments/ ./experiments/
COPY scenarios/ ./scenarios/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command with hot reload
CMD ["uv", "run", "uvicorn", "src.glitch_core.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production

# Copy source code
COPY src/ ./src/
COPY experiments/ ./experiments/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uv", "run", "uvicorn", "src.glitch_core.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 