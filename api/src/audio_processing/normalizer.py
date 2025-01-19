"""Audio level normalization."""

from typing import Dict, Optional

import numpy as np
from loguru import logger

from ..plugins import hookimpl
from ..structures.audio_schemas import NormConfig


class AudioNormalizer:
    """Handles audio normalization state for a single stream"""

    def __init__(self, config: Optional[NormConfig] = None):
        """Initialize normalizer.
        
        Args:
            config: Optional normalization configuration
        """
        self._config = config or NormConfig()
        self.int16_max = np.iinfo(np.int16).max
        self.samples_to_trim = int(self._config.chunk_trim_ms * self._config.sample_rate / 1000)

    def normalize(self, audio: np.ndarray, is_first: bool = True, is_last: bool = True) -> np.ndarray:
        """Convert audio data to int16 range."""
        if len(audio) == 0:
            raise ValueError("Audio data cannot be empty")
            
        # logger.debug(f"Input range: [{audio.min():.3f}, {audio.max():.3f}]")
            
        # Trim for non-final chunks
        if not is_last and len(audio) > self.samples_to_trim:
            audio = audio[:-self.samples_to_trim]
        
        # Convert to float32 if not already
        audio_float = audio.astype(np.float32)

        # Normalize to [-1, 1] range using max absolute value
        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float))

        # Scale to int16 range
        return (audio_float * self.int16_max).astype(np.int16)


# Module-level instance for convenience
_normalizer: Optional[AudioNormalizer] = None


def get_normalizer(config: Optional[NormConfig] = None) -> AudioNormalizer:
    """Get or create global normalizer instance.
    
    Args:
        config: Optional normalization configuration
        
    Returns:
        AudioNormalizer instance
    """
    global _normalizer
    if _normalizer is None:
        _normalizer = AudioNormalizer(config)
    return _normalizer


def normalize_audio(
    audio: np.ndarray,
    is_first: bool = True,
    is_last: bool = True
) -> np.ndarray:
    """Normalize audio using global normalizer instance.
    
    Args:
        audio: Audio samples
        is_first: Whether this is first chunk
        is_last: Whether this is last chunk
        
    Returns:
        Normalized audio
    """
    return get_normalizer().normalize(audio, is_first, is_last)