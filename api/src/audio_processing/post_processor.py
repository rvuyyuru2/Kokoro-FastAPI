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

    def process(
        self,
        audio: np.ndarray,
        is_first: bool = True,
        is_last: bool = True
    ) -> np.ndarray:
        """Apply fade effect.
        
        Args:
            audio: Audio samples
            is_first: Whether to fade in
            is_last: Whether to fade out
            
        Returns:
            Processed audio
        """
        if len(audio) < self._config.fade_samples * 2:
            return audio

        processed = audio.copy()
        
        # Create fade curve
        if self._config.fade_curve == "exponential":
            curve = np.exp(np.linspace(-4, 0, self._config.fade_samples))
        elif self._config.fade_curve == "cosine":
            curve = (1 - np.cos(np.linspace(0, np.pi, self._config.fade_samples))) / 2
        else:  # linear
            curve = np.linspace(0, 1, self._config.fade_samples)

        # Apply fades
        if is_first:
            processed[:self._config.fade_samples] *= curve
        if is_last:
            processed[-self._config.fade_samples:] *= curve[::-1]

        return processed

    def reset(self) -> None:
        """Reset fade effect state."""
        pass


class CompressorEffect(AudioEffect):
    """Dynamic range compression."""

    def __init__(self, config: EffectConfig):
        """Initialize compressor.
        
        Args:
            config: Effect configuration
        """
        self._config = config
        self._prev_gain = 1.0
        self._threshold = 10 ** (config.threshold_db / 20)
        self._attack_coef = np.exp(-1 / (config.attack_ms * 48))  # 48 = samples per ms
        self._release_coef = np.exp(-1 / (config.release_ms * 48))

    def process(
        self,
        audio: np.ndarray,
        is_first: bool = True,
        is_last: bool = True
    ) -> np.ndarray:
        """Apply compression.
        
        Args:
            audio: Audio samples
            is_first: Whether this is first chunk
            is_last: Whether this is last chunk
            
        Returns:
            Processed audio
        """
        # Calculate level
        level = np.abs(audio)
        mask = level > self._threshold
        
        # Calculate gain reduction
        gain = np.ones_like(audio)
        gain[mask] = (
            (level[mask] / self._threshold) **
            (1 / self._config.ratio - 1)
        )
        
        # Apply smoothing
        smoothed = np.zeros_like(gain)
        smoothed[0] = self._prev_gain
        
        for i in range(1, len(gain)):
            if gain[i] < smoothed[i-1]:
                smoothed[i] = (
                    self._attack_coef * smoothed[i-1] +
                    (1 - self._attack_coef) * gain[i]
                )
            else:
                smoothed[i] = (
                    self._release_coef * smoothed[i-1] +
                    (1 - self._release_coef) * gain[i]
                )
        
        # Save state
        self._prev_gain = smoothed[-1]
        
        return audio * smoothed

    def reset(self) -> None:
        """Reset compressor state."""
        self._prev_gain = 1.0


class AudioPostProcessor:
    """Handles audio post-processing effects."""

    def __init__(self, config: Optional[EffectConfig] = None):
        """Initialize post-processor.
        
        Args:
            config: Optional effect configuration
        """
        self._config = config or EffectConfig()
        self._effects: List[AudioEffect] = []
        self._initialize_effects()

    @hookimpl
    def pre_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Plugin hook for audio pre-processing.
        
        Args:
            audio: Audio samples
            
        Returns:
            Pre-processed audio
        """
        return audio

    @hookimpl
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Plugin hook for audio post-processing.
        
        Args:
            audio: Audio samples
            
        Returns:
            Post-processed audio
        """
        return audio

    def _initialize_effects(self) -> None:
        """Initialize default effects chain."""
        self._effects = [
            FadeEffect(self._config),
            CompressorEffect(self._config)
        ]

    def process(
        self,
        audio: np.ndarray,
        is_first: bool = True,
        is_last: bool = True
    ) -> np.ndarray:
        """Apply post-processing effects.
        
        Args:
            audio: Audio samples
            is_first: Whether this is first chunk
            is_last: Whether this is last chunk
            
        Returns:
            Processed audio
            
        Raises:
            ValueError: If audio is invalid
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Input must be numpy array")

        try:
            # Apply pre-processing
            processed = self.pre_process_audio(audio)

            # Apply effects chain
            for effect in self._effects:
                processed = effect.process(processed, is_first, is_last)

            # Apply post-processing
            processed = self.post_process_audio(processed)

            return processed

        except Exception as e:
            logger.error(f"Audio post-processing failed: {e}")
            raise

    def reset(self) -> None:
        """Reset all effects."""
        for effect in self._effects:
            effect.reset()


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