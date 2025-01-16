import time
from typing import AsyncGenerator, Optional
from loguru import logger

from .base_strategy import AudioStrategy, PlatformType
from ..audio import AudioNormalizer, AudioService
from ..text_processing import chunker, normalize_text
from ..tts_interface import TTSServiceInterface


class StreamingStrategy(AudioStrategy):
    """Strategy for streaming audio generation"""
    
    def __init__(self, tts_service: TTSServiceInterface):
        self.tts_service = tts_service
    
    @property
    def name(self) -> str:
        return "streaming"
    
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
        """Generate and yield audio chunks for real-time streaming"""
        try:
            stream_start = time.time()
            stream_normalizer = AudioNormalizer()

            # Input validation and preprocessing
            if not text:
                raise ValueError("Text is empty")
            preprocess_start = time.time()
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)
            logger.debug(
                f"Text preprocessing took: {(time.time() - preprocess_start)*1000:.1f}ms"
            )

            # Voice validation and loading
            voice_start = time.time()
            voice_path = await self.tts_service._find_voice(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
            voicepack = await self.tts_service._load_voice(str(voice_path), model)
            logger.debug(
                f"Voice loading took: {(time.time() - voice_start)*1000:.1f}ms"
            )

            # Process chunks as they're generated
            is_first = True
            chunks_processed = 0

            # Process chunks as they come from generator
            chunk_gen = chunker.split_text(text)
            current_chunk = next(chunk_gen, None)

            while current_chunk is not None:
                next_chunk = next(chunk_gen, None)  # Peek at next chunk
                chunks_processed += 1
                try:
                    # Process text and generate audio
                    language = self.tts_service._get_language_from_voice(voice)
                    model = await self.tts_service.model_manager.get_model(model)
                    try:
                        phonemes, tokens = model.process_text(current_chunk, language)
                        chunk_audio = model.generate_from_tokens(
                            tokens, voicepack, speed
                        )
                    finally:
                        self.tts_service.model_manager.release_model(model)

                    if chunk_audio is not None:
                        # Convert chunk with proper streaming header handling
                        chunk_bytes = AudioService.convert_audio(
                            chunk_audio,
                            24000,
                            output_format,
                            is_first_chunk=is_first,
                            normalizer=stream_normalizer,
                            is_last_chunk=(next_chunk is None),
                            stream=True
                        )

                        yield chunk_bytes
                        is_first = False
                    else:
                        logger.error(f"No audio generated for chunk: '{current_chunk}'")

                except Exception as e:
                    logger.error(
                        f"Failed to generate audio for chunk: '{current_chunk}'. Error: {str(e)}"
                    )

                current_chunk = next_chunk  # Move to next chunk

        except Exception as e:
            logger.error(f"Error in audio generation stream: {str(e)}")
            raise