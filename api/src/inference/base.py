"""Base interfaces for model inference."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch


from enum import Enum, auto


class ModelState(Enum):
    """Model lifecycle states."""
    UNINITIALIZED = auto()  # Initial state
    LOADED = auto()         # Model weights/file loaded
    WARMED_UP = auto()      # Completed warmup, ready for inference
    FAILED = auto()         # Failed state (load/warmup failed)
    UNLOADED = auto()       # Explicitly unloaded


class ModelBackend(ABC):
    """Abstract base class for model inference backends."""

    @abstractmethod
    async def load_model(self, path: str) -> None:
        """Load model from path.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    async def warmup(self) -> None:
        """Run model warmup.
        
        This should be called after load_model() to prepare the model
        for inference. Implementations should perform any necessary
        initialization like initial forward passes.
        
        Raises:
            RuntimeError: If warmup fails
        """
        pass

    @abstractmethod
    async def generate(
        self,
        tokens: List[int],
        voice: torch.Tensor,
        speed: float = 1.0,
        stream: Optional[torch.cuda.Stream] = None
    ) -> np.ndarray:
        """Generate audio from tokens.
        
        Args:
            tokens: Input token IDs
            voice: Voice embedding tensor
            speed: Speed multiplier
            stream: Optional CUDA stream for GPU inference
            
        Returns:
            Generated audio samples
            
        Raises:
            RuntimeError: If generation fails or model not ready
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free resources."""
        pass

    @property
    @abstractmethod
    def state(self) -> ModelState:
        """Get current model state.
        
        Returns:
            Current state of the model
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded.
        
        Returns:
            True if model is loaded (but may not be warmed up), False otherwise
        """
        return self.state in (ModelState.LOADED, ModelState.WARMED_UP)

    @property
    def is_ready(self) -> bool:
        """Check if model is ready for inference.
        
        Returns:
            True if model is loaded AND warmed up, False otherwise
        """
        return self.state == ModelState.WARMED_UP

    @property
    @abstractmethod
    def device(self) -> str:
        """Get device model is running on.
        
        Returns:
            Device string ('cpu' or 'cuda')
        """
        pass


class BaseModelBackend(ModelBackend):
    """Base implementation of model backend."""

    def __init__(self):
        """Initialize base backend."""
        self._state = ModelState.UNINITIALIZED
        self._device: str = "cpu"

    @property
    def state(self) -> ModelState:
        """Get current model state."""
        return self._state

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device

    def unload(self) -> None:
        """Unload model and free resources."""
        if self.is_loaded:
            self._cleanup_resources()
            self._state = ModelState.UNLOADED

    def _cleanup_resources(self) -> None:
        """Clean up model resources. Override in subclasses."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()