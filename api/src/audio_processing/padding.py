"""Audio padding and silence handling."""

from typing import Optional, Tuple

import numpy as np
from loguru import logger

from ..plugins import hookimpl
from ..structures.audio_schemas import PadConfig, PadStrategy


class AudioPadding:
    """Handles audio padding and silence."""

    def __init__(self, config: Optional[PadConfig] = None):
        """Initialize padding handler.
        
        Args:
            config: Optional padding configuration
        """
        self._config = config or PadConfig()

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

    def pad(
        self,
        audio: np.ndarray,
        pad_start: bool = True,
        pad_end: bool = True,
        duration_ms: float = 200.0
    ) -> np.ndarray:
        """Add padding to audio.
        
        Args:
            audio: Audio samples
            pad_start: Whether to pad start
            pad_end: Whether to pad end
            duration_ms: Padding duration in milliseconds
            
        Returns:
            Padded audio
            
        Raises:
            ValueError: If audio is invalid
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Input must be numpy array")

        try:
            # Apply pre-processing
            processed = self.pre_process_audio(audio)

            # Create padding
            pad_size = int(duration_ms * self._config.sample_rate / 1000)
            padding = self._create_padding(processed, pad_size)

            # Apply padding
            if pad_start and pad_end:
                processed = np.concatenate([padding, processed, padding])
            elif pad_start:
                processed = np.concatenate([padding, processed])
            elif pad_end:
                processed = np.concatenate([processed, padding])

            # Apply post-processing
            processed = self.post_process_audio(processed)

            return processed

        except Exception as e:
            logger.error(f"Audio padding failed: {e}")
            raise

    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, int, int]:
        """Trim silence from audio edges.
        
        Args:
            audio: Audio samples
            threshold: Optional custom threshold
            
        Returns:
            Tuple of:
                - Trimmed audio
                - Start index
                - End index
            
        Raises:
            ValueError: If audio is invalid
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Input must be numpy array")

        try:
            # Apply pre-processing
            processed = self.pre_process_audio(audio)

            # Find non-silent regions
            threshold = threshold or self._config.silence_threshold
            non_silent = np.where(np.abs(processed) > threshold)[0]
            
            if len(non_silent) == 0:
                # All silence
                return processed, 0, len(processed)
                
            # Get start/end indices
            start_idx = max(
                0,
                non_silent[0] - self._config.min_silence_samples
            )
            end_idx = min(
                len(processed),
                non_silent[-1] + self._config.min_silence_samples
            )
            
            # Trim audio
            processed = processed[start_idx:end_idx]

            # Apply post-processing
            processed = self.post_process_audio(processed)
            
            return processed, start_idx, end_idx

        except Exception as e:
            logger.error(f"Audio trimming failed: {e}")
            raise

    def _create_padding(self, audio: np.ndarray, size: int) -> np.ndarray:
        """Create padding based on strategy.
        
        Args:
            audio: Reference audio for some strategies
            size: Number of samples to generate
            
        Returns:
            Padding samples
        """
        if self._config.strategy == PadStrategy.SILENCE:
            return np.zeros(size)
            
        elif self._config.strategy == PadStrategy.NOISE:
            return (
                np.random.normal(0, self._config.noise_level, size)
                .astype(audio.dtype)
            )
            
        elif self._config.strategy == PadStrategy.FADE:
            # Create silence with fade
            padding = np.zeros(size)
            if size > self._config.fade_samples:
                fade = np.linspace(0, self._config.noise_level, self._config.fade_samples)
                padding[:self._config.fade_samples] = fade
                padding[-self._config.fade_samples:] = fade[::-1]
            return padding
            
        elif self._config.strategy == PadStrategy.REPEAT:
            # Repeat edge content with fade
            if len(audio) < size:
                repeats = int(np.ceil(size / len(audio)))
                padding = np.tile(audio, repeats)[:size]
            else:
                padding = audio[:size]
                
            # Apply fade
            if size > self._config.fade_samples:
                fade = np.linspace(1, 0, self._config.fade_samples)
                padding[:self._config.fade_samples] *= fade
                padding[-self._config.fade_samples:] *= fade[::-1]
                
            return padding
            
        else:
            raise ValueError(f"Unknown padding strategy: {self._config.strategy}")


# Module-level instance for convenience
_padding: Optional[AudioPadding] = None


def get_padding(config: Optional[PadConfig] = None) -> AudioPadding:
    """Get or create global padding instance.
    
    Args:
        config: Optional padding configuration
        
    Returns:
        AudioPadding instance
    """
    global _padding
    if _padding is None:
        _padding = AudioPadding(config)
    return _padding


def pad_audio(
    audio: np.ndarray,
    pad_start: bool = True,
    pad_end: bool = True,
    duration_ms: float = 200.0
) -> np.ndarray:
    """Pad audio using global padding instance.
    
    Args:
        audio: Audio samples
        pad_start: Whether to pad start
        pad_end: Whether to pad end
        duration_ms: Padding duration
        
    Returns:
        Padded audio
    """
    return get_padding().pad(audio, pad_start, pad_end, duration_ms)


def trim_silence(
    audio: np.ndarray,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, int, int]:
    """Trim silence using global padding instance.
    
    Args:
        audio: Audio samples
        threshold: Optional silence threshold
        
    Returns:
        Tuple of trimmed audio and indices
    """
    return get_padding().trim_silence(audio, threshold)