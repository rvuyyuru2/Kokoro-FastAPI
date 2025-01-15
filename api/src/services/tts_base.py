from abc import ABC, abstractmethod
from pathlib import Path
import threading
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger

from ..utils.paths import get_model_files, get_warmup_text
from .warmup import warmup_model


class TTSBaseModel(ABC):
    _instance = None
    _lock = threading.Lock()
    _device = None

    @classmethod
    async def setup(cls):
        """Initialize model and setup voices"""
        with cls._lock:
            # Set device
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                try:
                    # Test CUDA device
                    test_tensor = torch.zeros(1).cuda()
                    logger.info("CUDA test successful")
                    cls._device = "cuda"
                except Exception as e:
                    logger.error(f"CUDA test failed: {e}")
                    cls._device = "cpu"
            else:
                cls._device = "cpu"
            
            # Initialize model # TODO: Reconsider GPU ONNX
            is_onnx = cls._device == "cpu"
            suffix = ".onnx" if is_onnx else ".pth"
            
            # Find model files
            model_files = await get_model_files(suffix)
            if not model_files:
                raise RuntimeError(f"Could not find any {suffix} models in search paths")
                
            # Use first available model
            model_file = model_files[0]
            model_path = str(model_file)
            logger.info(f"Initializing model on {cls._device} using: {model_path}")
            model = cls.initialize(str(model_file.parent), model_path=model_path)
            if model is None:
                raise RuntimeError(f"Failed to initialize {cls._device.upper()} model")
            cls._instance = model

            # Load warmup text
            warmup_text, _ = await get_warmup_text()

            # Import here to avoid circular import
            # from .tts_service import TTSService

            # Create service and warm up
            voice_count = await warmup_model(warmup_text)

            logger.info("Model warm-up complete")

            # Return number of loaded voices
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
