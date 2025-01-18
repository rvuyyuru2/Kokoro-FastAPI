"""Pipeline interface definitions."""

from abc import ABC, abstractmethod
from typing import Protocol, Union


class GenerationStrategy(Protocol):
    """Protocol for generation strategies."""

    def generate(
        self,
        pipeline: 'Pipeline',
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav"
    ) -> Union[bytes, bytes]:
        """Generate audio using strategy.
        
        Args:
            pipeline: TTS pipeline
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            
        Returns:
            Audio data or chunks
        """
        ...


class Pipeline(ABC):
    """Base TTS pipeline interface."""

    @abstractmethod
    def process(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav"
    ) -> Union[bytes, bytes]:
        """Process text to speech.
        
        Args:
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            
        Returns:
            Audio data or chunks
        """
        ...