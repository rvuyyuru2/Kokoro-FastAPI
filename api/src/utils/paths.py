from pathlib import Path
from typing import List, Tuple
import aiofiles.os
from loguru import logger
from ..core.config import settings

async def _iterate_files(path: Path, suffix: str='') -> List[Path]:
    """Iterate over files in a directory"""
    files = []
    iterfiles = await aiofiles.os.scandir(path)
    for entry in iterfiles:
        if entry.is_file() and entry.name.endswith(suffix):
            files.append(Path(entry.path))
    return files

async def get_warmup_text() -> Tuple[str, bool]:
    """Get warmup text from don_quixote.txt"""
    warmup_path = Path(__file__).parent / "don_quixote.txt"
    try:
        async with aiofiles.open(warmup_path) as f:
            return await f.read(), True
    except Exception as e:
        logger.warning(f"Failed to load warmup text: {e}")
        return "This is a warmup text that will be split into chunks for processing.", False

async def get_model_files(suffix:str) -> List[Path]:
    """Get all model files from both directories"""
    files = []
    for path in [Path(settings.model_dir), Path("/app/defaults")]:
        if path.exists():
            files.extend(await _iterate_files(path, suffix))
    return files

async def get_voice_files() -> List[Path]:
    """Get all voice files from both directories"""
    files = []
    for path in [Path(settings.voices_dir), Path("/app/defaults")]:
        if path.exists():
            files.extend(await _iterate_files(path, ".pt"))
    return files
