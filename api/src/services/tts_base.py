import os
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import aiofiles.os
import numpy as np
import torch
from loguru import logger

from ..core.config import settings

async def scan_onnx_models(models_dir: str) -> Dict[str, str]:
    """Scan directory for ONNX models
    
    Args:
        models_dir: Directory to scan for .onnx files
        
    Returns:
        Dict mapping model names to full paths
    """
    models = {}
    try:
        logger.info(f"Scanning models directory: {models_dir}")
        if await aiofiles.os.path.exists(models_dir):
            logger.info(f"Directory exists, scanning for .onnx files")
            it = await aiofiles.os.scandir(models_dir)
            for entry in it:
                logger.info(f"Found entry: {entry.name}")
                if entry.name.endswith('.onnx'):
                    model_name = os.path.splitext(entry.name)[0]
                    models[model_name] = os.path.join(models_dir, entry.name)
                    logger.info(f"Added model: {model_name} -> {models[model_name]}")
        else:
            logger.warning(f"Models directory does not exist: {models_dir}")
    except Exception as e:
        logger.error(f"Error scanning models directory {models_dir}: {str(e)}")
    logger.info(f"Scan complete. Found models: {list(models.keys())}")
    return models


class TTSBaseModel(ABC):
    _instance = None
    _lock = threading.Lock()
    _device = None
    _available_models = {}
    _current_model_name = None
    _current_model_path = None
    VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voices")

    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get dictionary of available model names and paths"""
        return cls._available_models

    @classmethod
    def get_current_model(cls) -> Optional[str]:
        """Get name of currently loaded model"""
        return cls._current_model_name

    @classmethod
    async def reload_models(cls) -> Dict[str, str]:
        """Rescan for available models and update the model list
        
        Returns:
            Dict mapping model names to paths
        """
        with cls._lock:
            # Rescan for available models
            cls._available_models = await scan_onnx_models(settings.model_dir)
            
            # Add models from user models directory
            user_models = await scan_onnx_models(os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.models_dir))
            cls._available_models.update(user_models)
            
            logger.info(f"Updated available models: {list(cls._available_models.keys())}")
            return cls._available_models

    @classmethod
    async def setup(cls):
        """Initialize model and setup voices"""
        with cls._lock:
            # Set device to CPU for ONNX
            cls._device = "cpu"
            
            # Scan for available models
            cls._available_models = await scan_onnx_models(settings.model_dir)
            
            # Add models from user models directory if it exists
            user_models = await scan_onnx_models(os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.models_dir))
            cls._available_models.update(user_models)
            
            logger.info(f"Available models: {list(cls._available_models.keys())}")
            
            # Use default model path or fall back to first available
            model_name = os.path.splitext(settings.default_onnx_model)[0]
            if model_name not in cls._available_models:
                if not cls._available_models:
                    raise RuntimeError("No ONNX models found")
                model_name = next(iter(cls._available_models))
                logger.warning(f"Default model not found, using {model_name}")
            
            model_path = cls._available_models[model_name]
            cls._current_model_name = model_name
            cls._current_model_path = model_path
            
            logger.info(f"Initializing model on CPU")
            logger.info(f"Using model: {model_name}")
            logger.info(f"Model path: {model_path}")

            # Initialize model first
            model = cls.initialize(settings.model_dir, model_path=model_path)
            if model is None:
                raise RuntimeError(f"Failed to initialize {cls._device.upper()} model")
            cls._instance = model

            # Setup voices directory
            await aiofiles.os.makedirs(cls.VOICES_DIR, exist_ok=True)

            # Copy base voices to local directory
            base_voices_dir = os.path.join(settings.model_dir, settings.voices_dir)
            if await aiofiles.os.path.exists(base_voices_dir):
                it = await aiofiles.os.scandir(base_voices_dir)
                for entry in it:
                    if entry.name.endswith(".pt"):
                        voice_name = entry.name[:-3]
                        voice_path = os.path.join(cls.VOICES_DIR, entry.name)
                        if not await aiofiles.os.path.exists(voice_path):
                            try:
                                logger.info(
                                    f"Copying base voice {voice_name} to voices directory"
                                )
                                base_path = os.path.join(base_voices_dir, entry.name)
                                voicepack = torch.load(
                                    base_path,
                                    map_location=cls._device,
                                    weights_only=True,
                                )
                                torch.save(voicepack, voice_path)
                            except Exception as e:
                                logger.error(
                                    f"Error copying voice {voice_name}: {str(e)}"
                                )

            # Count voices in directory
            voice_count = 0
            it = await aiofiles.os.scandir(cls.VOICES_DIR)
            for entry in it:
                if entry.name.endswith(".pt"):
                    voice_count += 1

            # Now that model and voices are ready, do warmup
            try:
                warmup_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "core",
                    "don_quixote.txt",
                )
                async with aiofiles.open(warmup_path, mode='r') as f:
                    warmup_text = await f.read()
            except Exception as e:
                logger.warning(f"Failed to load warmup text: {e}")
                warmup_text = "This is a warmup text that will be split into chunks for processing."

            # Use warmup service after model is fully initialized
            from .warmup import WarmupService

            warmup = WarmupService()

            # Load and warm up voices
            loaded_voices = warmup.load_voices()
            await warmup.warmup_voices(warmup_text, loaded_voices)

            logger.info("Model warm-up complete")

            # Count voices in directory again after warmup
            voice_count = 0
            it = await aiofiles.os.scandir(cls.VOICES_DIR)
            for entry in it:
                if entry.name.endswith(".pt"):
                    voice_count += 1
            return voice_count

    @classmethod
    @abstractmethod
    def initialize(cls, model_dir: str, model_path: str = None):
        """Initialize the model"""
        pass

    @classmethod
    @abstractmethod
    def process_text(cls, text: str, language: str) -> Tuple[str, List[int]]:
        """Process text into phonemes and tokens

        Args:
            text: Input text
            language: Language code

        Returns:
            tuple[str, list[int]]: Phonemes and token IDs
        """
        pass

    @classmethod
    @abstractmethod
    def generate_from_text(
        cls, text: str, voicepack: torch.Tensor, language: str, speed: float
    ) -> Tuple[np.ndarray, str]:
        """Generate audio from text

        Args:
            text: Input text
            voicepack: Voice tensor
            language: Language code
            speed: Speed factor

        Returns:
            tuple[np.ndarray, str]: Generated audio samples and phonemes
        """
        pass

    @classmethod
    @abstractmethod
    def generate_from_tokens(
        cls, tokens: List[int], voicepack: torch.Tensor, speed: float
    ) -> np.ndarray:
        """Generate audio from tokens

        Args:
            tokens: Token IDs
            voicepack: Voice tensor
            speed: Speed factor

        Returns:
            np.ndarray: Generated audio samples
        """
        pass

    @classmethod
    def get_device(cls):
        """Get the current device"""
        if cls._device is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        return cls._device
