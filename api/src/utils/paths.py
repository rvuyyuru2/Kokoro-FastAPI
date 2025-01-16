from pathlib import Path
from typing import List, Tuple, Optional, Union
import aiofiles.os
from loguru import logger
from ..core.config import settings

# Fixed paths
MODEL_DIR = Path(settings.model_dir)
VOICE_DIR = Path(settings.voices_dir)

async def get_warmup_text() -> Tuple[str, bool]:
    """Get warmup text from don_quixote.txt"""
    warmup_path = Path(__file__).parent / "don_quixote.txt"
    try:
        async with aiofiles.open(warmup_path) as f:
            return await f.read(), True
    except Exception as e:
        logger.warning(f"Failed to load warmup text: {e}")
        return "This is a warmup text that will be split into chunks for processing.", False

async def get_model_file(pattern: Optional[str] = None) -> Path:
    """Get model file path
    
    Args:
        pattern: Optional pattern to match, e.g. "model.pth" for exact match
                or "*.pth" to find first .pth file. If not provided, returns
                first valid model found.
        
    Returns:
        Path to model file
        
    Raises:
        RuntimeError: If model file not found
    """
    if not MODEL_DIR.exists():
        raise RuntimeError("Model directory not found")
        
    if pattern:
        if '*' in pattern:
            # Pattern matching (e.g. "*.pth")
            suffix = pattern.split('*')[1]  # Get extension after *
            iterfiles = await aiofiles.os.scandir(MODEL_DIR)
            for entry in iterfiles:
                if entry.is_file() and entry.name.endswith(suffix):
                    return Path(entry.path)
            raise RuntimeError(f"No models found matching pattern: {pattern}")
        else:
            # Exact path provided
            path = MODEL_DIR / pattern
            if not path.exists():
                raise RuntimeError(f"Model not found: {pattern}")
            return path
        
    # No pattern provided - return first valid model
    iterfiles = await aiofiles.os.scandir(MODEL_DIR)
    for entry in iterfiles:
        if entry.is_file() and (entry.name.endswith('.onnx') or entry.name.endswith('.pth')):
            return Path(entry.path)
            
    raise RuntimeError("No models found")

async def get_voice_file(voice_name: Optional[str] = None) -> Union[Path, List[str]]:
    """Get voice file path or list all voices
    
    Args:
        voice_name: Optional voice name to find.
                   If not provided, returns list of all voice names.
        
    Returns:
        If voice_name provided: Path to voice file
        If no voice_name: List of available voice names
        
    Raises:
        RuntimeError: If voice directory not found or voice not found
    """
    if not VOICE_DIR.exists():
        raise RuntimeError("Voice directory not found")
        
    if voice_name:
        # Find specific voice
        path = VOICE_DIR / f"{voice_name}.pt"
        if not path.exists():
            raise RuntimeError(f"Voice not found: {voice_name}")
        return path
        
    # List all voices
    voices = []
    iterfiles = await aiofiles.os.scandir(VOICE_DIR)
    for entry in iterfiles:
        if entry.is_file() and entry.name.endswith('.pt'):
            voices.append(Path(entry.path).stem)
    return sorted(voices)
