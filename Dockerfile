# ==========================================
# STAGE 1: Builder
# ==========================================
FROM python:3.13-alpine AS builder

# Install build dependencies for numpy/scipy
RUN apk add --no-cache \
    gcc \
    g++ \
    gfortran \
    openblas-dev \
    lapack-dev \
    musl-dev

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies with optimizations
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv && \
    uv pip install --compile-bytecode -r pyproject.toml && \
    # Remove cache and unnecessary files
    find /app/.venv -name "*.pyc" -delete && \
    find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ==========================================
# STAGE 2: Runtime
# ==========================================
FROM python:3.13-alpine

# Install runtime dependencies only
RUN apk add --no-cache \
    libgomp \
    libstdc++ \
    openblas

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

# Copy only the venv
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ ./src/
COPY main.py .

CMD ["python", "main.py"]