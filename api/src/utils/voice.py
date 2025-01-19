"""Voice manipulation utilities."""

from typing import List

import torch
from loguru import logger

from ..core import paths


async def combine_voices(voices: List[str]) -> str:
    """Combine multiple voices.
    
    Args:
        voices: List of voice IDs
        
    Returns:
        Combined voice ID
        
    Raises:
        ValueError: If fewer than 2 voices
        RuntimeError: If combination fails
    """
    if len(voices) < 2:
        raise ValueError("At least 2 voices required")

    try:
        # Load voices
        voice_tensors = []
        voice_names = []

        for voice in voices:
            try:
                voice_path = await paths.get_voice_path(voice)
                if not voice_path:
                    raise ValueError(f"Voice not found: {voice}")

                # Load voice tensor
                tensor = await paths.load_voice_tensor(voice_path)
                voice_tensors.append(tensor)
                voice_names.append(voice)

            except Exception as e:
                logger.error(f"Failed to load voice {voice}: {e}")
                continue

        if len(voice_tensors) < 2:
            raise ValueError("Not enough valid voices to combine")

        # Combine voices
        combined_id = "_".join(voice_names)
        combined_tensor = torch.mean(torch.stack(voice_tensors), dim=0)
        
        # Save combined voice
        combined_path = await paths.get_voice_path(combined_id)
        await paths.save_voice_tensor(combined_tensor, combined_path)
        
        return combined_id

    except Exception as e:
        logger.error(f"Voice combination failed: {e}")
        raise