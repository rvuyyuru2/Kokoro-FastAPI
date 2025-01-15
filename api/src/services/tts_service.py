import io
import os
import re
import time
from functools import lru_cache
from typing import List, Optional, Tuple
from pathlib import Path

import aiofiles.os
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from loguru import logger

from ..core.config import settings
from ..utils.paths import get_voice_files
from .audio import AudioNormalizer, AudioService
from .text_processing import chunker, normalize_text
from .tts_model import TTSModel


class TTSService:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        self.model = TTSModel.get_instance()

    @staticmethod
    @lru_cache(maxsize=3)  # Cache up to 3 most recently used voices
    def _load_voice(voice_path: str) -> torch.Tensor:
        """Load and cache a voice model"""
        return torch.load(
            voice_path, map_location=TTSModel.get_device(), weights_only=True
        )

    async def _find_voice(self, voice_name: str) -> Optional[Path]:
        """Find a voice file by name"""
        voice_files = await get_voice_files()
        for voice_file in voice_files:
            if voice_file.stem == voice_name:
                return voice_file
        return None

    async def _generate_audio(
        self, text: str, voice: str, speed: float, stitch_long_output: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """Generate complete audio and return with processing time"""
        audio, processing_time = await self._generate_audio_internal(
            text, voice, speed, stitch_long_output
        )
        return audio, processing_time

    async def _generate_audio_internal(
        self, text: str, voice: str, speed: float, stitch_long_output: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """Generate audio and measure processing time"""
        start_time = time.time()

        try:
            # Normalize text once at the start
            if not text:
                raise ValueError("Text is empty after preprocessing")
            normalized = normalize_text(text)
            if not normalized:
                raise ValueError("Text is empty after preprocessing")
            text = str(normalized)

            # Find and load voice
            voice_path = await self._find_voice(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
            voicepack = self._load_voice(str(voice_path))

            # For non-streaming, preprocess all chunks first
            if stitch_long_output:
                # Preprocess all chunks to phonemes/tokens
                chunks_data = []
                for chunk in chunker.split_text(text):
                    try:
                        phonemes, tokens = TTSModel.process_text(chunk, voice[0])
                        chunks_data.append((chunk, tokens))
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
                        chunk_audio = TTSModel.generate_from_tokens(
                            tokens, voicepack, speed
                        )
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
            else:
                # Process single chunk
                phonemes, tokens = TTSModel.process_text(text, voice[0])
                audio = TTSModel.generate_from_tokens(tokens, voicepack, speed)

            processing_time = time.time() - start_time
            return audio, processing_time

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        speed: float,
        output_format: str = "wav",
        silent=False,
    ):
        """Generate and yield audio chunks as they're generated for real-time streaming"""
        try:
            stream_start = time.time()
            # Create normalizer for consistent audio levels
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
            voice_path = await self._find_voice(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
            voicepack = self._load_voice(str(voice_path))
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
                    phonemes, tokens = TTSModel.process_text(current_chunk, voice[0])
                    chunk_audio = TTSModel.generate_from_tokens(
                        tokens, voicepack, speed
                    )

                    if chunk_audio is not None:
                        # Convert chunk with proper streaming header handling
                        chunk_bytes = AudioService.convert_audio(
                            chunk_audio,
                            24000,
                            output_format,
                            is_first_chunk=is_first,
                            normalizer=stream_normalizer,
                            is_last_chunk=(next_chunk is None),  # Last if no next chunk
                            stream=True  # Ensure proper streaming format handling
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

    def _save_audio(self, audio: torch.Tensor, filepath: str):
        """Save audio to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        wavfile.write(filepath, 24000, audio)

    def _audio_to_bytes(self, audio: torch.Tensor) -> bytes:
        """Convert audio tensor to WAV bytes"""
        buffer = io.BytesIO()
        wavfile.write(buffer, 24000, audio)
        return buffer.getvalue()

    async def combine_voices(self, voices: List[str]) -> str:
        """Combine multiple voices into a new voice"""
        if len(voices) < 2:
            raise ValueError("At least 2 voices are required for combination")

        # Load voices
        t_voices: List[torch.Tensor] = []
        v_name: List[str] = []

        for voice in voices:
            try:
                voice_path = await self._find_voice(voice)
                if not voice_path:
                    raise ValueError(f"Voice not found: {voice}")
                voicepack = torch.load(
                    str(voice_path), map_location=TTSModel.get_device(), weights_only=True
                )
                t_voices.append(voicepack)
                v_name.append(voice)
            except Exception as e:
                raise ValueError(f"Failed to load voice {voice}: {str(e)}")

        # Combine voices
        try:
            f: str = "_".join(v_name)
            v = torch.mean(torch.stack(t_voices), dim=0)
            
            # Save to first available voice directory
            voice_files = await get_voice_files()
            if not voice_files:
                raise RuntimeError("No voice directories available")
            combined_path = str(voice_files[0].parent / f"{f}.pt")

            # Save combined voice
            try:
                torch.save(v, combined_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to save combined voice to {combined_path}: {str(e)}"
                )

            return f

        except Exception as e:
            if not isinstance(e, (ValueError, RuntimeError)):
                raise RuntimeError(f"Error combining voices: {str(e)}")
            raise

    async def list_voices(self) -> List[str]:
        """List all available voices"""
        try:
            voice_files = await get_voice_files()
            return sorted(voice_file.stem for voice_file in voice_files)
        except Exception as e:
            logger.error(f"Error listing voices: {str(e)}")
            return []
