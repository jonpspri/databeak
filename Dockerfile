# DataBeak MCP Server - HTTP Mode
# Multi-stage build for minimal production image

# Build stage - install dependencies with uv
FROM python:3.12-slim AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better layer caching
# README.md is required by pyproject.toml for package metadata
COPY pyproject.toml uv.lock README.md ./

# Create virtual environment and install production dependencies only
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/

# Install the project itself
RUN uv sync --frozen --no-dev


# Production stage - minimal runtime image
FROM python:3.12-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1000 databeak \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home databeak

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --from=builder /app/src /app/src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER databeak

# Expose HTTP port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the MCP server in HTTP mode
ENTRYPOINT ["python", "-m", "databeak.server"]
CMD ["--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
