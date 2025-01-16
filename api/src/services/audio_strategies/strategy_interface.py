from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Literal

import torch
import numpy as np

PlatformType = Literal["cpu", "gpu", "any"]

class AudioStrategy(ABC):
    """Base class for audio processing strategies"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass
    
    @property
    @abstractmethod
    def platform(self) -> PlatformType:
        """Platform requirement (cpu, gpu, any)"""
        pass
    
    @abstractmethod
    async def process_audio(
        self,
        text: str,
        voice: str,
        speed: float,
        output_format: str = "wav",
        model: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Process audio using the strategy's approach
        
        Args:
            text: Input text to convert to speech
            voice: Voice ID to use
            speed: Speech speed factor
            output_format: Output audio format
            model: Optional model name to use
            
        Yields:
            Audio data chunks
        """
        pass