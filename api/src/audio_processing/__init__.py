"""Audio processing module for TTS output."""

from typing import Optional
import numpy as np
from loguru import logger

from ..structures.audio_schemas import (
    AudioConfig,
    FormatConfig,
    NormConfig,
    PadConfig,
    EffectConfig,
    PadStrategy
)
from .converter import (
    AudioConverter,
    convert_audio,
    get_converter
)
from .normalizer import (
    AudioNormalizer,
    normalize_audio,
    get_normalizer
)
from .padding import (
    AudioPadding,
    pad_audio,
    trim_silence,
    get_padding
)
from .post_processor import (
    AudioEffect,
    AudioPostProcessor,
    CompressorEffect,
    FadeEffect,
    post_process_audio,
    get_post_processor
)


class AudioProcessor:
    """Complete audio processing pipeline."""

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize processor.
        
        Args:
            config: Optional configuration
        """
        self._config = config or AudioConfig()
        self._converter = AudioConverter(self._config.format)
        self._normalizer = AudioNormalizer(self._config.norm)
        self._padding = AudioPadding(self._config.pad)
        self._post_processor = AudioPostProcessor(self._config.effect)

    def process(
        self,
        audio: np.ndarray,
        format: str = "wav",
        add_padding: bool = True,
        normalize: bool = True,
        post_process: bool = True,
        is_first: bool = True,
        is_last: bool = True,
        stream: bool = False
    ) -> bytes:
        """Process audio through complete pipeline.
        
        Args:
            audio: Audio samples
            format: Output format
            add_padding: Whether to add padding
            normalize: Whether to normalize
            post_process: Whether to apply effects
            is_first: Whether this is first chunk
            is_last: Whether this is last chunk
            stream: Whether this is streaming
            
        Returns:
            Processed audio bytes
            
        Raises:
            ValueError: If processing fails
        """
        try:
            processed = audio

            # Optional padding
            if add_padding:
                processed = self._padding.pad(
                    processed,
                    pad_start=is_first,
                    pad_end=is_last
                )

            # Optional normalization
            if normalize:
                processed = self._normalizer.normalize(
                    processed,
                    is_first=is_first,
                    is_last=is_last
                )

            # Optional effects
            if post_process:
                processed = self._post_processor.process(
                    processed,
                    is_first=is_first,
                    is_last=is_last
                )

            # Convert format
            return self._converter.convert(
                processed,
                format,
                normalizer=self._normalizer if normalize else None,
                is_first_chunk=is_first,
                is_last_chunk=is_last,
                stream=stream
            )

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise

    def reset(self) -> None:
        """Reset all processor states."""
        self._post_processor.reset()


# Module-level instance for convenience
_processor: Optional[AudioProcessor] = None


def get_processor(config: Optional[AudioConfig] = None) -> AudioProcessor:
    """Get or create global processor instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        AudioProcessor instance
    """
    global _processor
    if _processor is None:
        _processor = AudioProcessor(config)
    return _processor


def process_audio(
    audio: np.ndarray,
    format: str = "wav",
    add_padding: bool = True,
    normalize: bool = True,
    post_process: bool = True,
    is_first: bool = True,
    is_last: bool = True,
    stream: bool = False
) -> bytes:
    """Process audio using global processor instance.
    
    Args:
        audio: Audio samples
        format: Output format
        add_padding: Whether to add padding
        normalize: Whether to normalize
        post_process: Whether to apply effects
        is_first: Whether this is first chunk
        is_last: Whether this is last chunk
        stream: Whether this is streaming
        
    Returns:
        Processed audio bytes
    """
    return get_processor().process(
        audio,
        format,
        add_padding,
        normalize,
        post_process,
        is_first,
        is_last,
        stream
    )


__all__ = [
    # Main processor
    "AudioProcessor",
    "AudioConfig",
    "process_audio",
    "get_processor",
    
    # Components
    "AudioConverter",
    "AudioNormalizer", 
    "AudioPadding",
    "AudioPostProcessor",
    "AudioEffect",
    
    # Effects
    "CompressorEffect",
    "FadeEffect",
    
    # Configurations
    "FormatConfig",
    "NormConfig", 
    "PadConfig",
    "EffectConfig",
    "PadStrategy",
    
    # Component functions
    "convert_audio",
    "normalize_audio",
    "pad_audio",
    "trim_silence",
    "post_process_audio",
    
    # Component getters
    "get_converter",
    "get_normalizer",
    "get_padding",
    "get_post_processor"
]