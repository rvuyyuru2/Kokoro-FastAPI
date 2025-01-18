"""Whole file strategy for TTS pipeline."""

from typing import Optional

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


class WholeFileStrategy(GenerationStrategy):
    """Strategy for complete file generation."""

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
    ) -> bytes:
        """Generate complete audio file.
        
        Args:
            pipeline: TTS pipeline
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            
        Returns:
            Audio file bytes
            
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

            # Generate audio chunks
            audio_chunks = []
            for tokens in token_sequences:
                try:
                    chunk_audio = self._model_manager.generate(
                        tokens,
                        voice_path,
                        speed
                    )
                    audio_chunks.append(chunk_audio)
                except Exception as e:
                    logger.error(f"Chunk generation failed: {e}")
                    continue

            if not audio_chunks:
                raise RuntimeError("No audio chunks generated")

            # Combine chunks
            complete_audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )

            # Process audio
            return self._audio_processor.process(
                complete_audio,
                format=format,
                add_padding=True,
                normalize=True,
                post_process=True,
                is_first_chunk=True,
                is_last_chunk=True,
                stream=False
            )

        except Exception as e:
            logger.error(f"Error in whole file generation: {str(e)}")
            raise

    def _combine_chunks(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Combine audio chunks.
        
        Args:
            chunks: List of audio chunks
            
        Returns:
            Combined audio
            
        Raises:
            ValueError: If no chunks
        """
        if not chunks:
            raise ValueError("No audio chunks to combine")

        # Add padding between chunks
        padded_chunks = []
        for i, chunk in enumerate(chunks):
            processed = self._audio_processor.process(
                chunk,
                format="wav",  # Temporary format for processing
                add_padding=True,
                normalize=True,
                post_process=True,
                is_first_chunk=(i == 0),
                is_last_chunk=(i == len(chunks) - 1),
                stream=False
            )
            padded_chunks.append(processed)

        # Combine chunks
        return np.concatenate(padded_chunks)