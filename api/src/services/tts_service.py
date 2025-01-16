import io
import os
import re
import time
from functools import lru_cache
from typing import List, Optional, Tuple, AsyncGenerator, Union
from pathlib import Path

import aiofiles.os
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from loguru import logger

from ..core.config import settings
from ..utils.paths import get_model_file, get_voice_file
from .audio import AudioNormalizer, AudioService
from .text_processing import chunker, normalize_text
from .tts_model import TTSModel
from .model_manager import ModelManager
from .tts_interface import TTSServiceInterface
from .audio_strategies.strategy_factory import AudioStrategyFactory


class TTSService(TTSServiceInterface):
    """Text-to-Speech service with concurrent model management"""
    
    # Singleton instance management
    _instance = None
    _model_manager = None
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        if TTSService._model_manager is None:
            TTSService._model_manager = ModelManager(TTSModel)
        self._model_manager = TTSService._model_manager
        self.strategy_factory = AudioStrategyFactory(self)

    @property
    def model_manager(self) -> ModelManager:
        return self._model_manager

    @classmethod
    async def shutdown_service(cls):
        """Gracefully shutdown the service and cleanup resources"""
        if cls._model_manager:
            await cls._model_manager.shutdown()
            cls._model_manager = None

    def _get_language_from_voice(self, voice: str) -> str:
        """Map voice prefix to phonemizer language code"""
        if voice.startswith('b'):
            return 'b'  # British English
        elif voice.startswith('j'):
            return 'j'  # Japanese
        return 'a'  # default to US English (voice starts with 'a' or other)

    @lru_cache(maxsize=3)  # Cache up to 3 most recently used voices
    async def _load_voice(self, voice_path: str, model_name: str = None) -> torch.Tensor:
        """Load and cache a voice model"""
        model = await self.model_manager.get_model(model_name)
        try:
            return torch.load(
                voice_path, map_location=model.get_device(), weights_only=True
            )
        finally:
            self.model_manager.release_model(model)

    async def _find_voice(self, voice_name: str) -> Optional[Path]:
        """Find a voice file by name"""
        try:
            return await get_voice_file(voice_name)
        except RuntimeError:
            return None

    async def _validate_model(self, model_name: str) -> None:
        """Validate model exists before starting audio generation
        
        Args:
            model_name: Name of model to validate
            
        Raises:
            RuntimeError: If model file not found
        """
        # Try exact model file
        try:
            await get_model_file(f"{model_name}.pth")
            return
        except RuntimeError:
            pass
            
        try:
            await get_model_file(f"{model_name}.onnx")
            return
        except RuntimeError:
            pass
            
        raise RuntimeError(f"Model not found: {model_name}")

    async def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        output_format: str = "wav",
        model: str = None,
    ) -> bytes:
        """Generate complete audio using the default stitch strategy"""
        if model:
            await self._validate_model(model)
            
        try:
            strategy = self.strategy_factory.get_strategy("default_stitch")
            async for audio_bytes in strategy.process_audio(
                text=text,
                voice=voice,
                speed=speed,
                output_format=output_format,
                model=model
            ):
                return audio_bytes  # Return the complete audio output
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
        model: str = None,
        strategy: str = "streaming"
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio using the specified strategy"""
        if model:
            await self._validate_model(model)
            
        try:
            audio_strategy = self.strategy_factory.get_strategy(strategy)
            async for chunk in audio_strategy.process_audio(
                text=text,
                voice=voice,
                speed=speed,
                output_format=output_format,
                model=model
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

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
                model = await self.model_manager.get_model()
                try:
                    voicepack = torch.load(
                        str(voice_path), map_location=model.get_device(), weights_only=True
                    )
                finally:
                    self.model_manager.release_model(model)
                t_voices.append(voicepack)
                v_name.append(voice)
            except Exception as e:
                raise ValueError(f"Failed to load voice {voice}: {str(e)}")

        # Combine voices
        try:
            f: str = "_".join(v_name)
            v = torch.mean(torch.stack(t_voices), dim=0)
            
            # Save combined voice
            try:
                combined_path = str(Path(settings.voices_dir) / f"{f}.pt")
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
            return await get_voice_file()  # No name = list all
        except RuntimeError as e:
            logger.error(f"Error listing voices: {str(e)}")
            return []
