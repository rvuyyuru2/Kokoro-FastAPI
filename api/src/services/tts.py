"""TTS service implementation."""

import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from ..core import paths

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
        try:
            self._config = config or ServiceConfig()
            self._model_manager = get_manager(self._config.model)
            self._audio_processor = get_audio_processor(self._config.audio)
            logger.info("TTS service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            raise RuntimeError(f"Service initialization failed: {e}")

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
            logger.info("Processing text input")
            try:
                token_sequences = process_text(text)
                if not token_sequences:
                    raise ValueError("Text processing produced no valid chunks")
                logger.debug(f"Generated {len(token_sequences)} token sequences")
            except Exception as e:
                logger.error(f"Text processing failed: {e}")
                raise ValueError(f"Text processing failed: {e}")

            # Load voice
            voice_path = await self._get_voice_path(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")

            # Generate audio
            audio_chunks = []
            for tokens in token_sequences:
                chunk_audio = await self._model_manager.generate(
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
            logger.info("Processing text input for streaming")
            try:
                token_sequences = process_text(text)
                if not token_sequences:
                    raise ValueError("Text processing produced no valid chunks")
                logger.debug(f"Generated {len(token_sequences)} token sequences for streaming")
            except Exception as e:
                logger.error(f"Text processing failed for streaming: {e}")
                raise ValueError(f"Text processing failed: {e}")

            # Load voice
            voice_start = time.time()
            voice_path = await self._get_voice_path(voice)
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
                    chunk_audio = await self._model_manager.generate(
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
                    voice_path = await self._get_voice_path(voice)
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

    async def _get_voice_path(self, voice: str) -> str:
        """Get path to voice file.
        
        Args:
            voice: Voice ID
            
        Returns:
            Voice file path
            
        Raises:
            ValueError: If voice not found
        """
        try:
            return await paths.get_voice_path(voice)
        except RuntimeError as e:
            raise ValueError(str(e))

    async def shutdown(self):
        """Cleanup resources on shutdown."""
        logger.info("Shutting down TTS service")
        self._model_manager.unload_all()
        logger.info("Resources cleaned up")

    def _validate_model(self) -> None:
        """Validate model is loaded.
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._model_manager.get_backend().is_loaded:
            raise RuntimeError("Model not loaded")

    async def list_voices(self) -> List[str]:
        """List all available voices.
        
        Returns:
            List of voice IDs
        """
        try:
            return await paths.list_voices()
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []


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