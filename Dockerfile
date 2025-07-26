FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv
RUN uv sync --frozen

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

# Run the application
CMD ["uv", "run", "uvicorn", "src.glitch_core.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 