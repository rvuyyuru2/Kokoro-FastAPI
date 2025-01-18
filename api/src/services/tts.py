"""TTS service implementation."""

import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Tuple

import aiofiles.os
import numpy as np
import torch
from loguru import logger

from ..structures.audio_schemas import AudioConfig
from ..audio_processing import (
    AudioProcessor,
    get_processor as get_audio_processor
)
from ..core.config import settings
from ..inference import ModelConfig, ModelManager, get_manager
from ..plugins import hookimpl
from ..text_processing import process_text


@dataclass
class ServiceConfig:
    """TTS service configuration."""
    
    model: ModelConfig = ModelConfig()
    audio: AudioConfig = AudioConfig()
    voices_dir: str = "voices"
    output_dir: Optional[str] = None


class TTSService:
    """Text-to-speech service."""

    def __init__(self, config: Optional[ServiceConfig] = None):
        """Initialize service.
        
        Args:
            config: Optional service configuration
        """
        self._config = config or ServiceConfig()
        self._model_manager = get_manager(self._config.model)
        self._audio_processor = get_audio_processor(self._config.audio)
        self._voices_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            self._config.voices_dir
        )

    @hookimpl
    def pre_process_text(self, text: str) -> str:
        """Plugin hook for text pre-processing.
        
        Args:
            text: Raw input text
            
        Returns:
            Pre-processed text
        """
        return text

    @hookimpl
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Plugin hook for audio post-processing.
        
        Args:
            audio: Audio samples
            
        Returns:
            Post-processed audio
        """
        return audio

    async def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav"
    ) -> bytes:
        """Generate complete audio file.
        
        Args:
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            output_format: Output format
            
        Returns:
            Audio file bytes
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If generation fails
        """
        try:
            # Apply pre-processing hook
            text = self.pre_process_text(text)

            # Process text
            token_sequences = process_text(text)
            if not token_sequences:
                raise ValueError("No valid text chunks")

            # Load voice
            voice_path = self._get_voice_path(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")

            # Generate audio
            audio_chunks = []
            for tokens in token_sequences:
                chunk_audio = self._model_manager.generate(
                    tokens,
                    voice_path,
                    speed
                )
                audio_chunks.append(chunk_audio)

            # Combine chunks
            audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )

            # Apply post-processing hook
            audio = self.post_process_audio(audio)

            # Process audio
            return self._audio_processor.process(
                audio,
                format=output_format,
                stream=False
            )

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise

    async def generate_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav"
    ) -> AsyncIterator[bytes]:
        """Generate streaming audio.
        
        Args:
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            output_format: Output format
            
        Yields:
            Audio chunks
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If generation fails
        """
        try:
            # Track timing
            stream_start = time.time()

            # Apply pre-processing hook
            text = self.pre_process_text(text)

            # Process text
            token_sequences = process_text(text)
            if not token_sequences:
                raise ValueError("No valid text chunks")

            # Load voice
            voice_start = time.time()
            voice_path = self._get_voice_path(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
            logger.debug(
                f"Voice loading took: {(time.time() - voice_start)*1000:.1f}ms"
            )

            # Generate chunks
            is_first = True
            total_chunks = len(token_sequences)

            for i, tokens in enumerate(token_sequences):
                try:
                    # Generate audio
                    chunk_audio = self._model_manager.generate(
                        tokens,
                        voice_path,
                        speed
                    )

                    # Apply post-processing hook
                    chunk_audio = self.post_process_audio(chunk_audio)

                    # Process audio
                    chunk_bytes = self._audio_processor.process(
                        chunk_audio,
                        format=output_format,
                        is_first=is_first,
                        is_last=(i == total_chunks - 1),
                        stream=True
                    )

                    yield chunk_bytes
                    is_first = False

                except Exception as e:
                    logger.error(f"Chunk generation failed: {e}")
                    continue

            # Log total time
            total_time = time.time() - stream_start
            logger.info(
                f"Stream completed in {total_time:.2f}s "
                f"({total_chunks} chunks)"
            )

        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise

    async def combine_voices(self, voices: List[str]) -> str:
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
                    voice_path = self._get_voice_path(voice)
                    if not voice_path:
                        raise ValueError(f"Voice not found: {voice}")

                    tensor = torch.load(
                        voice_path,
                        map_location="cpu",
                        weights_only=True
                    )
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
            combined_path = os.path.join(
                self._voices_dir,
                f"{combined_id}.pt"
            )

            # Save combined voice
            torch.save(combined_tensor, combined_path)
            return combined_id

        except Exception as e:
            logger.error(f"Voice combination failed: {e}")
            raise

    async def list_voices(self) -> List[str]:
        """List available voices.
        
        Returns:
            List of voice IDs
        """
        voices = []
        try:
            it = await aiofiles.os.scandir(self._voices_dir)
            for entry in it:
                if entry.name.endswith(".pt"):
                    voices.append(entry.name[:-3])
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
        return sorted(voices)

    def _get_voice_path(self, voice: str) -> Optional[str]:
        """Get path to voice file.
        
        Args:
            voice: Voice ID
            
        Returns:
            Voice file path or None if not found
        """
        path = os.path.join(self._voices_dir, f"{voice}.pt")
        return path if os.path.exists(path) else None


# Module-level instance
_service: Optional[TTSService] = None


def get_service(config: Optional[ServiceConfig] = None) -> TTSService:
    """Get or create global service instance.
    
    Args:
        config: Optional service configuration
        
    Returns:
        TTSService instance
    """
    global _service
    if _service is None:
        _service = TTSService(config)
    return _service