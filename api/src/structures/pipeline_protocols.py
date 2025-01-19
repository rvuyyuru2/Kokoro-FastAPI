"""Pipeline interface definitions."""

from abc import ABC, abstractmethod
from typing import Protocol, AsyncIterator, Optional, Union


class Pipeline(ABC):
    """Base TTS pipeline interface."""

    @abstractmethod
    async def get_voice_path(self, voice: str) -> Optional[str]:
        """Get path to voice file.
        
        Args:
            voice: Voice ID
            
        Returns:
            Voice file path or None if not found
        """
        ...

    @abstractmethod
    async def cleanup_voice(self, voice_path: str) -> None:
        """Cleanup voice resources.
        
        Args:
            voice_path: Path to voice file
        """
        ...

    @abstractmethod
    async def process(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav",
        stream: bool = False
    ) -> Union[AsyncIterator[bytes], bytes]:
        """Process text to speech.
        
        Args:
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            stream: Whether to stream output
            
        Returns:
            Audio chunks for streaming pipeline, complete audio for whole file pipeline
        """
        ...