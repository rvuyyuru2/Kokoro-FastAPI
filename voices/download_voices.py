#!/usr/bin/env python3
import os
import sys
import requests
from pathlib import Path
from typing import List

def download_file(url: str, output_dir: Path) -> None:
    """Download a file from URL to the specified directory."""
    filename = os.path.basename(url)
    output_path = output_dir / filename
    
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main(custom_models: List[str] = None):
    # Always use top-level voices directory relative to project root
    project_root = Path(__file__).resolve().parent.parent  # Go up one level from voices/
    voices_dir = project_root / 'voices'
    voices_dir.mkdir(exist_ok=True)
    
    # Default voice models if no arguments provided
    default_models = [
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_bella.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_irulan.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_nicole.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_sarah.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/af_sky.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_adam.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_gurney.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/am_michael.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bf_emma.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bf_isabella.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bm_george.pt",
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/bm_lewis.pt"
    ]
    
    # Use provided models or default
    models_to_download = custom_models if custom_models else default_models
    
    for model_url in models_to_download:
        try:
            download_file(model_url, voices_dir)
        except Exception as e:
            print(f"Error downloading {model_url}: {e}")

if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)