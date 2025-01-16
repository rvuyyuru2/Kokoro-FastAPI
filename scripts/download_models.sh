#!/bin/bash

# Ensure models and voices directories exist
mkdir -p models voices

# Function to download a file
download_file() {
    local url="$1"
    local filename=$(basename "$url")
    echo "Downloading $filename..."
    curl -L "$url" -o "models/$filename"
}

# Base model
download_file "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.onnx"

# Voice models
VOICE_MODELS=(
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_bella.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_irulan.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_nicole.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_sarah.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_sky.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_adam.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_gurney.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_michael.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bf_emma.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bf_isabella.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bm_george.pt"
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bm_lewis.pt"
)

# Download all voice models
for model in "${VOICE_MODELS[@]}"; do
    download_file "$model"
done

echo "Model download complete!"