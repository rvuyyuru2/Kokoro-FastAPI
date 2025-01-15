FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    espeak-ng \
    git \
    libsndfile1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root user and directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models \
            /app/defaults \
            /app/voices && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Download default model files
RUN cd /app/defaults && \
    curl -L -o kokoro-v0_19.pth "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.pth"

RUN cd /app/defaults && \
    curl -L -o af_bella.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af.pt" && \
    curl -L -o af_bella.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_bella.pt" && \
    curl -L -o af_irulan.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_irulan.pt" && \
    curl -L -o af_nicole.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_nicole.pt" && \
    curl -L -o af_sarah.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_sarah.pt" && \
    curl -L -o af_sky.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_sky.pt" && \
    curl -L -o am_adam.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_adam.pt" && \
    curl -L -o am_gurney.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_gurney.pt" && \
    curl -L -o am_michael.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_michael.pt" && \
    curl -L -o bf_emma.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bf_emma.pt" && \
    curl -L -o bf_isabella.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bf_isabella.pt" && \
    curl -L -o bm_george.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bm_george.pt" && \
    curl -L -o bm_lewis.pt "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bm_lewis.pt"

WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv sync --extra gpu --no-install-project

# Copy project files
COPY --chown=appuser:appuser api ./api

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/models
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_LINK_MODE=copy

# Run FastAPI server
CMD ["uv", "run", "python", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880", "--log-level", "debug"]
