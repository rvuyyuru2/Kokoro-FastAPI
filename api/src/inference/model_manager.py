"""Model management and caching."""

import os
from typing import Dict, Optional, Union

import torch
from loguru import logger

from .base import ModelBackend
from .cpu import CPUBackend
from .gpu import GPUBackend
from ..core.config import settings
from ..structures.model_schemas import ModelConfig


class ModelManager:
    """Manages model loading and inference across backends."""

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model manager.
        
        Args:
            config: Optional configuration
        """
        self._config = config or ModelConfig()
        self._backends: Dict[str, ModelBackend] = {}
        self._voice_cache: Dict[str, torch.Tensor] = {}
        self._current_backend: Optional[str] = None
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize available backends."""
        if settings.use_gpu and torch.cuda.is_available():
            try:
                self._backends['gpu'] = GPUBackend()
                self._current_backend = 'gpu'
                logger.info("Initialized GPU backend")
            except Exception as e:
                logger.error(f"Failed to initialize GPU backend: {e}")
                # Fallback to CPU if GPU fails
                self._initialize_cpu()
        else:
            self._initialize_cpu()

    def _initialize_cpu(self) -> None:
        """Initialize CPU backend."""
        try:
            self._backends['cpu'] = CPUBackend()
            self._current_backend = 'cpu'
            logger.info("Initialized CPU backend")
        except Exception as e:
            logger.error(f"Failed to initialize CPU backend: {e}")
            raise RuntimeError("No backends available")

    def get_backend(self, backend_type: Optional[str] = None) -> ModelBackend:
        """Get specified backend.
        
        Args:
            backend_type: Backend type ('cpu' or 'gpu'), uses default if None
            
        Returns:
            Model backend instance
            
        Raises:
            ValueError: If backend type is invalid
            RuntimeError: If no backends are available
        """
        if not self._backends:
            raise RuntimeError("No backends available")

        if backend_type is None:
            backend_type = self._current_backend
        
        if backend_type not in self._backends:
            raise ValueError(f"Invalid backend type: {backend_type}")

        return self._backends[backend_type]

    def load_model(
        self,
        model_path: str,
        backend_type: Optional[str] = None
    ) -> None:
        """Load model on specified backend.
        
        Args:
            model_path: Path to model file
            backend_type: Backend to load on, uses default if None
            
        Raises:
            RuntimeError: If model loading fails
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")

        backend = self.get_backend(backend_type)
        try:
            backend.load_model(model_path)
            logger.info(f"Loaded model on {backend_type or self._current_backend} backend")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def load_voice(
        self,
        voice_path: str,
        backend_type: Optional[str] = None
    ) -> torch.Tensor:
        """Load and cache voice tensor.
        
        Args:
            voice_path: Path to voice file
            backend_type: Backend to load for, uses default if None
            
        Returns:
            Voice tensor
            
        Raises:
            RuntimeError: If voice loading fails
        """
        # Check cache first
        if self._config.cache_voices and voice_path in self._voice_cache:
            return self._voice_cache[voice_path]

        try:
            # Load voice tensor
            backend = self.get_backend(backend_type)
            device = backend.device
            voice = torch.load(voice_path, map_location=device)

            # Cache if enabled
            if self._config.cache_voices:
                self._manage_voice_cache()
                self._voice_cache[voice_path] = voice
                logger.debug(f"Cached voice: {voice_path}")

            return voice

        except Exception as e:
            raise RuntimeError(f"Failed to load voice: {e}")

    def _manage_voice_cache(self) -> None:
        """Manage voice cache size."""
        if len(self._voice_cache) >= self._config.voice_cache_size:
            # Remove oldest voice
            oldest = next(iter(self._voice_cache))
            del self._voice_cache[oldest]
            logger.debug(f"Removed from voice cache: {oldest}")

    def generate(
        self,
        tokens: list[int],
        voice_path: str,
        speed: float = 1.0,
        backend_type: Optional[str] = None
    ) -> torch.Tensor:
        """Generate audio using specified backend.
        
        Args:
            tokens: Input token IDs
            voice_path: Path to voice file
            speed: Speed multiplier
            backend_type: Backend to use, uses default if None
            
        Returns:
            Generated audio tensor
            
        Raises:
            RuntimeError: If generation fails
        """
        backend = self.get_backend(backend_type)
        if not backend.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Load voice
            voice = self.load_voice(voice_path, backend_type)
            
            # Generate audio
            return backend.generate(tokens, voice, speed)
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def unload_all(self) -> None:
        """Unload models from all backends."""
        for backend in self._backends.values():
            backend.unload()
        self._voice_cache.clear()
        logger.info("Unloaded all models and cleared voice cache")

    @property
    def available_backends(self) -> list[str]:
        """Get list of available backends.
        
        Returns:
            List of backend names
        """
        return list(self._backends.keys())

    @property
    def current_backend(self) -> str:
        """Get current default backend.
        
        Returns:
            Backend name
        """
        return self._current_backend

    @property
    def voice_cache_info(self) -> Dict[str, int]:
        """Get voice cache statistics.
        
        Returns:
            Dictionary with cache info
        """
        return {
            'size': len(self._voice_cache),
            'max_size': self._config.voice_cache_size
        }


# Module-level instance
_manager: Optional[ModelManager] = None


def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get or create global model manager instance.
    
    Args:
        config: Optional model configuration
        
    Returns:
        ModelManager instance
    """
    global _manager
    if _manager is None:
        _manager = ModelManager(config)
    return _manager