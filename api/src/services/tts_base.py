from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
from typing import List, Tuple, Optional

import numpy as np
import torch
from loguru import logger

from ..utils.paths import get_model_files


class TTSBaseModel(ABC):
    """Base class for TTS models with resource management"""
    
    def __init__(self):
        self._device = None
        self._model = None
        self._lock = asyncio.Lock()
        self._busy = False

    async def setup(self, model_path: Optional[str] = None):
        """Initialize model"""
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
            
            # Initialize model
            is_onnx = self._device == "cpu"
            required_suffix = ".onnx" if is_onnx else ".pth"
            
            if model_path is not None:
                # When specific model requested, validate exact match with correct extension
                requested_path = Path(model_path)
                model_files = await get_model_files(required_suffix)
                matching_model = next((f for f in model_files if f.stem == requested_path.stem), None)
                
                if not matching_model:
                    available_models = ", ".join(f.stem for f in model_files)
                    error_msg = f"Model '{requested_path.stem}' not found with {required_suffix} extension. Available models: {available_models}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                model_path = str(matching_model)
            else:
                # Default model selection only if no model specified
                model_files = await get_model_files(required_suffix)
                if not model_files:
                    logger.error(f"Could not find any {required_suffix} models in search paths")
                    raise RuntimeError(f"Could not find any {required_suffix} models in search paths")
                model_path = str(model_files[0])

            logger.info(f"Initializing model on {self._device} using: {model_path}")
            model_dir = str(Path(model_path).parent)
            self._model = self.initialize(model_dir, model_path=model_path)
            if self._model is None:
                raise RuntimeError(f"Failed to initialize {self._device.upper()} model")
            
            logger.info("Model initialization complete")

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
