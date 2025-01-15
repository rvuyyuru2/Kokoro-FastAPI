FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    curl -L -o kokoro-v0_19.onnx "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.onnx"
    # curl -L -o kokoro-v0_19.onnx "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.onnx" && \

RUN cd /app/defaults && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_bella.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_irulan.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_nicole.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_sarah.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_sky.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_adam.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_gurney.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_michael.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bf_emma.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bf_isabella.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bm_george.pt" && \
    curl -LO "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bm_lewis.pt"

WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv sync --extra cpu --no-install-project

# Copy project files
COPY --chown=appuser:appuser api ./api

# Install project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra cpu

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/models
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV ONNX_NUM_THREADS=8
ENV ONNX_INTER_OP_THREADS=4
ENV ONNX_EXECUTION_MODE=sequential
ENV ONNX_OPTIMIZATION_LEVEL=all
ENV ONNX_MEMORY_PATTERN=true
ENV ONNX_ARENA_EXTEND_STRATEGY=kNextPowerOfTwo

# Run FastAPI server
CMD ["uv", "run", "python", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8880", "--log-level", "debug"]
