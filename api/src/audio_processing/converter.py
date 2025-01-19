"""Audio format conversion."""

import io
from typing import Dict, Optional

import numpy as np
import soundfile as sf
from loguru import logger

from ..structures.audio_schemas import FormatConfig
from .normalizer import AudioNormalizer


class AudioConverter:
    """Handles audio format conversion."""

    SUPPORTED_FORMATS = {"wav", "mp3", "opus", "flac", "pcm"}

    def __init__(self, config: Optional[FormatConfig] = None):
        """Initialize converter.
        
        Args:
            config: Optional format configuration
        """
        self._config = config or FormatConfig()
        self._sample_rate = 24000  # Fixed sample rate for TTS

    def convert(
        self,
        audio: np.ndarray,
        format: str = "wav",
        normalizer: Optional[AudioNormalizer] = None,
        is_first_chunk: bool = True,
        is_last_chunk: bool = True,
        stream: bool = False
    ) -> bytes:
        """Convert audio array to specified format.
        
        Args:
            audio: Audio samples
            format: Output format
            normalizer: Optional audio normalizer
            is_first_chunk: Whether this is first chunk
            is_last_chunk: Whether this is last chunk
            stream: Whether this is streaming
            
        Returns:
            Audio data in specified format
            
        Raises:
            ValueError: If format is unsupported
        """
        format = format.lower()
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Format {format} not supported. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        try:
            # Normalize if needed
            if normalizer is not None:
                audio = normalizer.normalize(
                    audio,
                    is_first=is_first_chunk,
                    is_last=is_last_chunk
                )

            # Convert format
            if stream:
                return self._convert_stream(
                    audio,
                    format,
                    is_first_chunk,
                    is_last_chunk
                )
            else:
                return self._convert_complete(audio, format)

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise

    def _convert_stream(
        self,
        audio: np.ndarray,
        format: str,
        is_first: bool,
        is_last: bool
    ) -> bytes:
        """Convert audio for streaming.
        
        Args:
            audio: Audio samples
            format: Output format
            is_first: Whether this is first chunk
            is_last: Whether this is last chunk
            
        Returns:
            Audio chunk bytes
        """
        buffer = io.BytesIO()

        if format == "pcm":
            # Raw PCM data
            return audio.tobytes()

        elif format == "wav":
            # WAV with headers only on first chunk
            if is_first:
                sf.write(
                    buffer,
                    audio,
                    self._sample_rate,
                    format="WAV",
                    subtype="PCM_16"
                )
                return buffer.getvalue()
            else:
                return audio.tobytes()

        else:
            # Other formats need complete chunks
            settings = self._config.get_format_settings(format)
            if format == "mp3":
                sf.write(
                    buffer,
                    audio,
                    self._sample_rate,
                    format="MP3",
                    **settings
                )
            elif format == "opus":
                sf.write(
                    buffer,
                    audio,
                    self._sample_rate,
                    format="OGG",
                    subtype="OPUS",
                    **settings
                )
            elif format == "flac":
                sf.write(
                    buffer,
                    audio,
                    self._sample_rate,
                    format="FLAC",
                    subtype="PCM_16",
                    **settings
                )

            buffer.seek(0)
            return buffer.getvalue()

    def _convert_complete(self, audio: np.ndarray, format: str) -> bytes:
        """Convert complete audio file.
        
        Args:
            audio: Audio samples
            format: Output format
            
        Returns:
            Audio file bytes
        """
        buffer = io.BytesIO()
        settings = self._config.get_format_settings(format)

        if format == "pcm":
            return audio.tobytes()

        elif format == "wav":
            sf.write(
                buffer,
                audio,
                self._sample_rate,
                format="WAV",
                subtype="PCM_16"
            )

        elif format == "mp3":
            sf.write(
                buffer,
                audio,
                self._sample_rate,
                format="MP3",
                **settings
            )

        elif format == "opus":
            sf.write(
                buffer,
                audio,
                self._sample_rate,
                format="OGG",
                subtype="OPUS",
                **settings
            )

        elif format == "flac":
            sf.write(
                buffer,
                audio,
                self._sample_rate,
                format="FLAC",
                subtype="PCM_16",
                **settings
            )

        buffer.seek(0)
        return buffer.getvalue()


# Module-level instance for convenience
_converter: Optional[AudioConverter] = None


def get_converter(config: Optional[FormatConfig] = None) -> AudioConverter:
    """Get or create global converter instance.
    
    Args:
        config: Optional format configuration
        
    Returns:
        AudioConverter instance
    """
    global _converter
    if _converter is None:
        _converter = AudioConverter(config)
    return _converter


def convert_audio(
    audio: np.ndarray,
    format: str = "wav",
    normalizer: Optional[AudioNormalizer] = None,
    is_first_chunk: bool = True,
    is_last_chunk: bool = True,
    stream: bool = False
) -> bytes:
    """Convert audio using global converter instance.
    
    Args:
        audio: Audio samples
        format: Output format
        normalizer: Optional normalizer
        is_first_chunk: Whether this is first chunk
        is_last_chunk: Whether this is last chunk
        stream: Whether this is streaming
        
    Returns:
        Converted audio data
    """
    return get_converter().convert(
        audio,
        format,
        normalizer,
        is_first_chunk,
        is_last_chunk,
        stream
    )