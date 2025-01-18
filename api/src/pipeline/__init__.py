"""TTS pipeline module."""

from typing import Union

from .base import (
    GenerationStrategy,
    Pipeline,
    StrategyPipeline,
    create_pipeline
)
from .streaming import StreamingStrategy
from .whole_file import WholeFileStrategy


def create_streaming_pipeline(
    voices_dir: str = "voices"
) -> Pipeline:
    """Create streaming pipeline.
    
    Args:
        voices_dir: Voice directory path
        
    Returns:
        Pipeline instance
    """
    return create_pipeline(
        strategy="streaming",
        voices_dir=voices_dir
    )


def create_whole_file_pipeline(
    voices_dir: str = "voices"
) -> Pipeline:
    """Create whole file pipeline.
    
    Args:
        voices_dir: Voice directory path
        
    Returns:
        Pipeline instance
    """
    return create_pipeline(
        strategy="whole_file",
        voices_dir=voices_dir
    )


def process_text(
    text: str,
    voice: str,
    speed: float = 1.0,
    format: str = "wav",
    stream: bool = False,
    voices_dir: str = "voices"
) -> Union[bytes, bytes]:
    """Process text to speech.
    
    Args:
        text: Input text
        voice: Voice ID
        speed: Speed multiplier
        format: Output format
        stream: Whether to stream output
        voices_dir: Voice directory path
        
    Returns:
        Audio data or chunks
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If processing fails
    """
    pipeline = (
        create_streaming_pipeline(voices_dir)
        if stream else
        create_whole_file_pipeline(voices_dir)
    )
    return pipeline.process(text, voice, speed, format)


__all__ = [
    # Base classes
    "GenerationStrategy",
    "Pipeline",
    "StrategyPipeline",
    
    # Strategies
    "StreamingStrategy",
    "WholeFileStrategy",
    
    # Factory functions
    "create_pipeline",
    "create_streaming_pipeline",
    "create_whole_file_pipeline",
    
    # Convenience function
    "process_text"
]