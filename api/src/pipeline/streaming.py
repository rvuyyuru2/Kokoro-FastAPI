"""Streaming pipeline implementation."""

from typing import AsyncIterator, List, Union

import numpy as np
from loguru import logger

from .base import BasePipeline
from ..text_processing import process_text


class StreamingPipeline(BasePipeline):
    """Streaming pipeline implementation."""

    async def process(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav",
        stream: bool = False
    ) -> AsyncIterator[bytes]:
        """Process text to speech with streaming.
        
        Args:
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            stream: Whether to stream output (ignored, always streams)
            
        Returns:
            Audio chunk iterator
        """
        # Get voice path
        voice_path = await self.get_voice_path(voice)
        if not voice_path:
            raise ValueError(f"Voice not found: {voice}")

        try:
            # Preprocess text
            processed_text = self._apply_text_preprocessing(text)

            # Process text into token sequences
            token_sequences = await process_text(processed_text)
            if not token_sequences:
                raise ValueError("Text processing failed to generate tokens")

            # Stream chunks
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

                    # Apply post-processing
                    chunk_audio = self._apply_audio_postprocessing(chunk_audio)

                    # Process audio
                    chunk_bytes = self._audio_processor.process(
                        chunk_audio,
                        format=format,
                        add_padding=True,
                        normalize=True,
                        post_process=True,
                        is_first=is_first,
                        is_last=(i == total_chunks - 1),
                        stream=True
                    )

                    yield chunk_bytes
                    is_first = False

                except Exception as e:
                    logger.error(f"Chunk generation failed: {e}")
                    yield b""  # Return empty chunk on error

        finally:
            await self.cleanup_voice(voice_path)