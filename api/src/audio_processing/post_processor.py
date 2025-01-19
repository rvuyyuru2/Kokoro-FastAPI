"""Audio post-processing effects."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from ..plugins import hookimpl
from ..structures.audio_schemas import EffectConfig


class AudioEffect(ABC):
    """Base class for audio effects."""

    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        is_first: bool = True,
        is_last: bool = True
    ) -> np.ndarray:
        """Process audio through effect.
        
        Args:
            audio: Audio samples
            is_first: Whether this is first chunk
            is_last: Whether this is last chunk
            
        Returns:
            Processed audio
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset effect state."""
        pass


class FadeEffect(AudioEffect):
    """Applies fade in/out effects."""

    def __init__(self, config: EffectConfig):
        """Initialize fade effect.
        
        Args:
            config: Effect configuration
        """
        self._config = config

    def process(self, audio: np.ndarray, is_first: bool = True, is_last: bool = True) -> np.ndarray:
        """Apply fade effect."""
        if len(audio) < self._config.fade_samples * 2:
            return audio

        # Ensure float32 throughout
        processed = audio.astype(np.float32)
        
        # Create fade curve in float32
        curve = np.linspace(0, 1, self._config.fade_samples, dtype=np.float32)

        # Apply fades
        if is_first:
            processed[:self._config.fade_samples] = processed[:self._config.fade_samples] * curve
        if is_last:
            processed[-self._config.fade_samples:] = processed[-self._config.fade_samples:] * curve[::-1]

        return processed.astype(audio.dtype)

    def reset(self) -> None:
        """Reset fade effect state."""
        pass


class AudioPostProcessor:
    """Handles audio post-processing effects."""

    def __init__(self, config: Optional[EffectConfig] = None):
        """Initialize post-processor.
        
        Args:
            config: Optional effect configuration
        """
        self._config = config or EffectConfig()
        self._fade = FadeEffect(self._config)

    @hookimpl
    def pre_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Plugin hook for audio pre-processing."""
        return audio

    @hookimpl
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Plugin hook for audio post-processing."""
        return audio

    def process(self, audio: np.ndarray, is_first: bool = True, is_last: bool = True) -> np.ndarray:
        """Apply post-processing effects."""
        if not isinstance(audio, np.ndarray):
            raise ValueError("Input must be numpy array")

        try:
            processed = self.pre_process_audio(audio)
            processed = self._fade.process(processed, is_first, is_last)
            return self.post_process_audio(processed)
        except Exception as e:
            logger.error(f"Audio post-processing failed: {e}")
            raise

    def reset(self) -> None:
        """Reset effects."""
        self._fade.reset()


# Module-level instance for convenience
_post_processor: Optional[AudioPostProcessor] = None


def get_post_processor(config: Optional[EffectConfig] = None) -> AudioPostProcessor:
    """Get or create global post-processor instance.
    
    Args:
        config: Optional effect configuration
        
    Returns:
        AudioPostProcessor instance
    """
    global _post_processor
    if _post_processor is None:
        _post_processor = AudioPostProcessor(config)
    return _post_processor


def post_process_audio(
    audio: np.ndarray,
    is_first: bool = True,
    is_last: bool = True
) -> np.ndarray:
    """Post-process audio using global instance.
    
    Args:
        audio: Audio samples
        is_first: Whether this is first chunk
        is_last: Whether this is last chunk
        
    Returns:
        Processed audio
    """
    return get_post_processor().process(audio, is_first, is_last)