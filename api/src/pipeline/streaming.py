"""Streaming strategy for TTS pipeline."""

from typing import Iterator, Optional

import numpy as np
from loguru import logger

from ..audio_processing import (
    AudioConfig,
    AudioProcessor,
    get_processor as get_audio_processor
)
from ..inference import ModelManager, get_manager
from ..text_processing import process_text
from .base import GenerationStrategy, Pipeline


class StreamingStrategy(GenerationStrategy):
    """Strategy for streaming audio generation."""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        audio_processor: Optional[AudioProcessor] = None
    ):
        """Initialize strategy.
        
        Args:
            model_manager: Optional model manager
            audio_processor: Optional audio processor
        """
        self._model_manager = model_manager or get_manager()
        self._audio_processor = audio_processor or get_audio_processor()

    def generate(
        self,
        pipeline: Pipeline,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav"
    ) -> Iterator[bytes]:
        """Generate audio in streaming chunks.
        
        Args:
            pipeline: TTS pipeline
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            
        Returns:
            Iterator yielding audio chunks
        
        Yields:
            bytes: Audio chunks in specified format
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If generation fails
        """
        try:
            # Input validation
            if not text:
                raise ValueError("Empty input text")
            if not voice:
                raise ValueError("No voice specified")

            # Process text into token sequences
            token_sequences = process_text(text)
            if not token_sequences:
                raise ValueError("No valid text chunks")

            # Get voice path
            voice_path = pipeline.get_voice_path(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")

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

                    # Process audio
                    chunk_bytes = self._audio_processor.process(
                        chunk_audio,
                        format=format,
                        add_padding=True,
                        normalize=True,
                        post_process=True,
                        is_first_chunk=is_first,
                        is_last_chunk=(i == total_chunks - 1),
                        stream=True
                    )

                    yield chunk_bytes
                    is_first = False

                except Exception as e:
                    logger.error(f"Chunk generation failed: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            raise

    def _process_chunk(
        self,
        chunk_audio: np.ndarray,
        format: str,
        is_first: bool,
        is_last: bool
    ) -> bytes:
        """Process audio chunk.
        
        Args:
            chunk_audio: Audio samples
            format: Output format
            is_first: Whether this is first chunk
            is_last: Whether this is last chunk
            
        Returns:
            Processed audio bytes
        """
        return self._audio_processor.process(
            chunk_audio,
            format=format,
            add_padding=True,
            normalize=True,
            post_process=True,
            is_first_chunk=is_first,
            is_last_chunk=is_last,
            stream=True
        )