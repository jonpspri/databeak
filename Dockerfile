# Multi-stage Docker build for DataBeak HTTP streaming server
FROM python:3.11-slim as builder

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY requirements.txt ./

# Install dependencies in virtual environment
RUN uv venv && \
    uv pip install -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Create non-root user for security
RUN groupadd -r databeak && useradd -r -g databeak -u 1001 databeak

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=databeak:databeak /app/.venv /app/.venv

# Copy application code
COPY --chown=databeak:databeak pyproject.toml requirements.txt README.md ./
COPY --chown=databeak:databeak src/ src/

# Make sure the virtual environment is in the path
ENV PATH="/app/.venv/bin:$PATH"

# Install the project in the virtual environment
RUN uv pip install -e .

# Switch to non-root user
USER databeak

# Set environment variables for production
ENV DATABEAK_LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for HTTP streaming
EXPOSE 8000

# Health check endpoint (basic connectivity test)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command runs HTTP streaming server
CMD ["databeak", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
