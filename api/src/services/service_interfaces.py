"""Interfaces to prevent circular imports"""
from typing import Protocol, Optional, AsyncGenerator, List
from pathlib import Path
import torch


class TTSServiceProtocol(Protocol):
    """Protocol defining TTSService interface to prevent circular imports"""
    
    async def _find_voice(self, voice_name: str) -> Optional[Path]:
        """Find a voice file by name"""
        ...
        
    async def _load_voice(self, voice_path: str, model_name: str = None) -> torch.Tensor:
        """Load and cache a voice model"""
        ...
        
    def _get_language_from_voice(self, voice: str) -> str:
        """Map voice prefix to phonemizer language code"""
        ...
        
    @property
    def model_manager(self):
        """Get the model manager"""
        ...