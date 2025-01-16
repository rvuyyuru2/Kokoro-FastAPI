from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import torch

from .model_manager import ModelManager


class TTSServiceInterface(ABC):
    """Interface defining TTS service capabilities needed by strategies"""
    
    @property
    @abstractmethod
    def model_manager(self) -> ModelManager:
        """Get the model manager instance"""
        pass
    
    @abstractmethod
    def _get_language_from_voice(self, voice: str) -> str:
        """Map voice prefix to phonemizer language code"""
        pass
    
    @abstractmethod
    async def _load_voice(self, voice_path: str, model_name: Optional[str] = None) -> torch.Tensor:
        """Load and cache a voice model"""
        pass
    
    @abstractmethod
    async def _find_voice(self, voice_name: str) -> Optional[Path]:
        """Find a voice file by name"""
        pass