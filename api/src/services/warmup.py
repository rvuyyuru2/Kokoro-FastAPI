from pathlib import Path
from typing import List, Tuple

import torch
from loguru import logger

from ..utils.paths import get_voice_files


async def warmup_model(warmup_text: str) -> int:
    """Warm up the TTS model and voices"""
    # Import here to avoid circular import
    from .tts_service import TTSService
    service = TTSService()

    # Get all voices sorted by filename length (shorter names first, usually base voices)
    voice_files = sorted(
        await get_voice_files(), 
        key=lambda p: len(p.name)
    )

    # Load first voice for warmup
    n_voices_cache = 1
    loaded_voices = []
    for voice_path in voice_files[:n_voices_cache]:
        try:
            voice_path_str = str(voice_path)
            voicepack = service._load_voice(voice_path_str)
            loaded_voices.append((voice_path.stem, voicepack))
            logger.info(f"Loaded voice {voice_path.stem}")
        except Exception as e:
            logger.error(f"Failed to load voice {voice_path.name}: {e}")
            continue

    if not loaded_voices:
        logger.warning("No voices loaded for warmup")
        return 0

    # Run warmup inference
    for voice_name, _ in loaded_voices:
        try:
            logger.info(f"Running warmup inference on voice {voice_name}")
            async for _ in service.generate_audio_stream(
                warmup_text, voice_name, 1.0, "pcm"
            ):
                pass  # Process all chunks to properly warm up
            logger.info(f"Completed warmup for voice {voice_name}")
        except Exception as e:
            logger.warning(f"Warmup failed for voice {voice_name}: {e}")

    return len(loaded_voices)
