"""Whole file pipeline implementation."""

from typing import AsyncIterator, List, Union

import numpy as np
from loguru import logger

from .base import BasePipeline
from ..text_processing import process_text


class WholeFilePipeline(BasePipeline):
    """Complete file generation pipeline."""

    async def process(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav",
        stream: bool = False
    ) -> bytes:
        """Process text to speech as complete file.
        
        Args:
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            stream: Whether to stream output (ignored, always returns complete file)
            
        Returns:
            Complete audio file bytes
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

            # Generate audio chunks
            audio_chunks = []
            for tokens in token_sequences:
                try:
                    # Generate audio
                    chunk_audio = await self._model_manager.generate(
                        tokens,
                        voice_path,
                        speed
                    )
                    
                    # Apply post-processing
                    chunk_audio = self._apply_audio_postprocessing(chunk_audio)
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

            # Process final audio
            return self._audio_processor.process(
                complete_audio,
                format=format,
                add_padding=True,
                normalize=True,
                post_process=True,
                is_first=True,
                is_last=True,
                stream=False
            )

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise
        finally:
            await self.cleanup_voice(voice_path)