"""Audio level normalization."""

from typing import Dict, Optional

import numpy as np
from loguru import logger

from ..plugins import hookimpl
from ..structures.audio_schemas import NormConfig


class AudioNormalizer:
    """Handles audio level normalization."""

    def __init__(self, config: Optional[NormConfig] = None):
        """Initialize normalizer.
        
        Args:
            config: Optional normalization configuration
        """
        self._config = config or NormConfig()
        self._running_stats = None
        self._reset_stats()

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

    def normalize(
        self,
        audio: np.ndarray,
        is_first: bool = True,
        is_last: bool = True
    ) -> np.ndarray:
        """Normalize audio levels.
        
        Args:
            audio: Audio samples
            is_first: Whether this is first chunk
            is_last: Whether this is last chunk
            
        Returns:
            Normalized audio
            
        Raises:
            ValueError: If audio is invalid
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Input must be numpy array")

        try:
            # Apply pre-processing
            audio = self.pre_process_audio(audio)

            # Reset stats for new stream
            if is_first:
                self._reset_stats()

            # Update running statistics
            self._update_stats(audio)

            # Apply normalization
            if is_last:
                # Final normalization using complete stats
                normalized = self._normalize_complete(audio)
            else:
                # Progressive normalization for streaming
                normalized = self._normalize_progressive(audio)

            # Apply post-processing
            normalized = self.post_process_audio(normalized)

            # Apply chunk trimming if needed
            if not is_last and len(normalized) > self._config.trim_samples:
                normalized = normalized[:-self._config.trim_samples]

            return normalized

        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            raise

    def _reset_stats(self) -> None:
        """Reset running statistics."""
        self._running_stats = {
            'sum': 0.0,
            'sum_squares': 0.0,
            'peak': 0.0,
            'count': 0,
            'windows': []
        }

    def _update_stats(self, audio: np.ndarray) -> None:
        """Update running statistics.
        
        Args:
            audio: Audio samples
        """
        # Basic statistics
        self._running_stats['sum'] += np.sum(audio)
        self._running_stats['sum_squares'] += np.sum(audio ** 2)
        self._running_stats['peak'] = max(
            self._running_stats['peak'],
            np.max(np.abs(audio))
        )
        self._running_stats['count'] += len(audio)

        # Window-based analysis
        for i in range(0, len(audio), self._config.window_size):
            window = audio[i:i + self._config.window_size]
            if len(window) == self._config.window_size:
                rms = np.sqrt(np.mean(window ** 2))
                self._running_stats['windows'].append(rms)

    def _normalize_complete(self, audio: np.ndarray) -> np.ndarray:
        """Normalize using complete statistics.
        
        Args:
            audio: Audio samples
            
        Returns:
            Normalized audio
        """
        # Calculate RMS level
        mean = self._running_stats['sum'] / self._running_stats['count']
        mean_square = (
            self._running_stats['sum_squares'] / self._running_stats['count']
        )
        rms = np.sqrt(mean_square - mean ** 2)

        # Convert to dB
        current_db = 20 * np.log10(max(rms, self._config.min_level))
        
        # Calculate gain
        gain = 10 ** ((self._config.target_db - current_db) / 20)
        
        # Apply gain with peak limiting
        normalized = audio * gain
        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized /= peak

        return normalized

    def _normalize_progressive(self, audio: np.ndarray) -> np.ndarray:
        """Progressive normalization for streaming.
        
        Args:
            audio: Audio samples
            
        Returns:
            Normalized audio
        """
        # Use window-based analysis for more stable streaming
        if self._running_stats['windows']:
            window_rms = np.mean(self._running_stats['windows'])
            current_db = 20 * np.log10(max(window_rms, self._config.min_level))
            gain = 10 ** ((self._config.target_db - current_db) / 20)
        else:
            # Fallback to basic stats if no windows yet
            mean = self._running_stats['sum'] / self._running_stats['count']
            mean_square = (
                self._running_stats['sum_squares'] / self._running_stats['count']
            )
            rms = np.sqrt(mean_square - mean ** 2)
            current_db = 20 * np.log10(max(rms, self._config.min_level))
            gain = 10 ** ((self._config.target_db - current_db) / 20)

        # Apply gain with peak limiting
        normalized = audio * gain
        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized /= peak

        return normalized

    @property
    def stats(self) -> Dict:
        """Get current normalization statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self._running_stats:
            return {}
            
        return {
            'mean': (
                self._running_stats['sum'] / self._running_stats['count']
                if self._running_stats['count'] > 0
                else 0.0
            ),
            'peak': self._running_stats['peak'],
            'window_count': len(self._running_stats['windows']),
            'sample_count': self._running_stats['count']
        }


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