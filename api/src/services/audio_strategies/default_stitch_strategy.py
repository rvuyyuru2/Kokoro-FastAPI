from typing import AsyncGenerator, Optional
import numpy as np
from loguru import logger

from .base_strategy import AudioStrategy, PlatformType
from ..audio import AudioNormalizer, AudioService
from ..text_processing import chunker, normalize_text
from ..tts_interface import TTSServiceInterface


class DefaultStitchStrategy(AudioStrategy):
    """Strategy for stitching together audio chunks from sequential processing"""
    
    def __init__(self, tts_service: TTSServiceInterface):
        self.tts_service = tts_service
    
    @property
    def name(self) -> str:
        return "default_stitch"
    
    @property
    def platform(self) -> PlatformType:
        return "any"
        
    async def process_audio(
        self,
        text: str,
        voice: str,
        speed: float,
        output_format: str = "wav",
        model: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Generate complete audio by processing chunks sequentially and stitching"""
        try:
            # Normalize text once at the start
            if not text:
                raise ValueError("Text is empty")
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Find and load voice
            voice_path = await self.tts_service._find_voice(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
            voicepack = await self.tts_service._load_voice(str(voice_path), model)

            # Preprocess all chunks to phonemes/tokens
            chunks_data = []
            for chunk in chunker.split_text(text):
                try:
                    language = self.tts_service._get_language_from_voice(voice)
                    model = await self.tts_service.model_manager.get_model(model)
                    try:
                        phonemes, tokens = model.process_text(chunk, language)
                        chunks_data.append((chunk, tokens))
                    finally:
                        self.tts_service.model_manager.release_model(model)
                except Exception as e:
                    logger.error(
                        f"Failed to process chunk: '{chunk}'. Error: {str(e)}"
                    )
                    continue

            if not chunks_data:
                raise ValueError("No chunks were processed successfully")

            # Generate audio for all chunks
            audio_chunks = []
            for chunk, tokens in chunks_data:
                try:
                    model = await self.tts_service.model_manager.get_model(model)
                    try:
                        chunk_audio = model.generate_from_tokens(
                            tokens, voicepack, speed
                        )
                    finally:
                        self.tts_service.model_manager.release_model(model)
                    if chunk_audio is not None:
                        audio_chunks.append(chunk_audio)
                    else:
                        logger.error(f"No audio generated for chunk: '{chunk}'")
                except Exception as e:
                    logger.error(
                        f"Failed to generate audio for chunk: '{chunk}'. Error: {str(e)}"
                    )
                    continue

            if not audio_chunks:
                raise ValueError("No audio chunks were generated successfully")

            # Concatenate all chunks
            audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )

            # Convert to desired format and yield as single chunk
            normalizer = AudioNormalizer()
            audio_bytes = AudioService.convert_audio(
                audio,
                24000,
                output_format,
                is_first_chunk=True,
                normalizer=normalizer,
                is_last_chunk=True,
                stream=False
            )
            yield audio_bytes

        except Exception as e:
            logger.error(f"Error in sequential audio generation: {str(e)}")
            raise