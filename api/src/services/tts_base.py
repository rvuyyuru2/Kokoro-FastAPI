from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
from typing import List, Tuple, Optional

import numpy as np
import torch
from loguru import logger

from ..utils.paths import get_model_file


class TTSBaseModel(ABC):
    """Base class for TTS models with resource management"""
    
    def __init__(self):
        self._device = None
        self._model = None
        self._lock = asyncio.Lock()
        self._busy = False

    async def setup(self, model_name: Optional[str] = None):
        """Initialize model with optional specific model name
        
        Args:
            model_name: Name of model to load (must be exact match)
            
        Raises:
            RuntimeError: If model initialization fails
        """
        async with self._lock:
            # Set device
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                try:
                    # Test CUDA device
                    test_tensor = torch.zeros(1).cuda()
                    logger.info("CUDA test successful")
                    self._device = "cuda"
                except Exception as e:
                    logger.error(f"CUDA test failed: {e}")
                    self._device = "cpu"
            else:
                self._device = "cpu"
            
            # CUDA = .pth, CPU = .onnx - no exceptions
            suffix = ".pth" if self._device == "cuda" else ".onnx"
            
            try:
                # Get exact model file if name provided, otherwise first valid model
                if model_name:
                    model_path = await get_model_file(f"{model_name}{suffix}")
                else:
                    model_path = await get_model_file(f"*{suffix}")
                    
                logger.info(f"Initializing model on {self._device} using: {model_path}")
                model_dir = str(model_path.parent)
                self._model = self.initialize(model_dir, model_path=str(model_path))
                if self._model is None:
                    raise RuntimeError(f"Failed to initialize {self._device.upper()} model")
                
                logger.info("Model initialization complete")
                
            except Exception as e:
                raise RuntimeError(f"Model initialization failed: {str(e)}")

    @abstractmethod
    def initialize(self, model_dir: str, model_path: str = None):
        """Initialize the model"""
        pass

    @abstractmethod
    def process_text(self, text: str, language: str) -> Tuple[str, List[int]]:
        """Process text into phonemes and tokens

        Args:
            text: Input text
            language: Language code

        Returns:
            tuple[str, list[int]]: Phonemes and token IDs
        """
        pass

    @abstractmethod
    def generate_from_text(
        self, text: str, voicepack: torch.Tensor, language: str, speed: float
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

    @abstractmethod
    def generate_from_tokens(
        self, tokens: List[int], voicepack: torch.Tensor, speed: float
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

    def get_device(self):
        """Get the current device"""
        if self._device is None:
            raise RuntimeError("Model not initialized. Call setup() first.")
        return self._device

    async def acquire(self):
        """Mark model as busy"""
        await self._lock.acquire()
        self._busy = True

    def release(self):
        """Mark model as available"""
        self._busy = False
        self._lock.release()

    def cache_clear(self):
        """Clear any cached data"""
        if hasattr(self, '_load_voice'):
            self._load_voice.cache_clear()
        # Clear any other caches specific to subclasses
        
    @property
    def is_busy(self) -> bool:
        """Check if model is currently processing"""
        return self._busy
