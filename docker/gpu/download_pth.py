#!/usr/bin/env python3
import os
import sys
import requests
from pathlib import Path
from typing import List

def download_file(url: str, output_dir: Path) -> None:
    """Download a file from URL to the specified directory."""
    filename = os.path.basename(url)
    if not filename.endswith('.pth'):
        print(f"Warning: {filename} is not a .pth file")
        return
        
    output_path = output_dir / filename
    
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def find_project_root() -> Path:
    """Find project root by looking for api directory."""
    current = Path(__file__).resolve()
    while current.parent != current:  # Stop at root directory
        if (current.parent / 'api').is_dir():
            return current.parent
        current = current.parent
    raise RuntimeError("Could not find project root (no api directory found)")

def main(custom_models: List[str] = None):
    # Find project root and ensure models directory exists
    project_root = find_project_root()
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Default PTH model if no arguments provided
    default_models = [
        "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.0/kokoro-v0_19.pth"
    ]
    
    # Use provided models or default
    models_to_download = custom_models if custom_models else default_models
    
    for model_url in models_to_download:
        try:
            download_file(model_url, models_dir)
        except Exception as e:
            print(f"Error downloading {model_url}: {e}")

if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)