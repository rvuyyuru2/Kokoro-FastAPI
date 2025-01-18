#!/bin/bash

# Ensure models directory exists
mkdir -p models

# Function to download a file
download_file() {
    local url="$1"
    local filename=$(basename "$url")
    echo "Downloading $filename..."
    curl -L "$url" -o "models/$filename"
}

# Default ONNX model if no arguments provided
DEFAULT_MODELS=(
    "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.onnx"
)

# Use provided models or default
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Download all models
for model in "${MODELS[@]}"; do
    download_file "$model"
done

echo "ONNX model download complete!"