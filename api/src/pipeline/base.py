"""Base classes for TTS pipeline."""

import os
from typing import Optional, Union

from loguru import logger

from ..structures.pipeline_protocols import Pipeline, GenerationStrategy
from ..audio_processing import (
    AudioProcessor,
    get_processor as get_audio_processor
)
from ..inference import ModelManager, get_manager


class BasePipeline(Pipeline):
    """Base TTS pipeline implementation."""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        audio_processor: Optional[AudioProcessor] = None,
        voices_dir: str = "voices"
    ):
        """Initialize pipeline.
        
        Args:
            model_manager: Optional model manager
            audio_processor: Optional audio processor
            voices_dir: Voice directory path
        """
        self._model_manager = model_manager or get_manager()
        self._audio_processor = audio_processor or get_audio_processor()
        self._voices_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            voices_dir
        )

    def get_voice_path(self, voice: str) -> Optional[str]:
        """Get path to voice file.
        
        Args:
            voice: Voice ID
            
        Returns:
            Voice file path or None if not found
        """
        path = os.path.join(self._voices_dir, f"{voice}.pt")
        return path if os.path.exists(path) else None

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
        raise NotImplementedError


class StrategyPipeline(BasePipeline):
    """Pipeline using generation strategy."""

    def __init__(
        self,
        strategy: GenerationStrategy,
        model_manager: Optional[ModelManager] = None,
        audio_processor: Optional[AudioProcessor] = None,
        voices_dir: str = "voices"
    ):
        """Initialize pipeline.
        
        Args:
            strategy: Generation strategy
            model_manager: Optional model manager
            audio_processor: Optional audio processor
            voices_dir: Voice directory path
        """
        super().__init__(model_manager, audio_processor, voices_dir)
        self._strategy = strategy

    def process(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav"
    ) -> Union[bytes, bytes]:
        """Process text to speech using strategy.
        
        Args:
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            
        Returns:
            Audio data or chunks
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If processing fails
        """
        try:
            return self._strategy.generate(
                self,
                text,
                voice,
                speed,
                format
            )
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise


def create_pipeline(
    strategy: str = "whole_file",
    model_manager: Optional[ModelManager] = None,
    audio_processor: Optional[AudioProcessor] = None,
    voices_dir: str = "voices"
) -> Pipeline:
    """Create pipeline with strategy.
    
    Args:
        strategy: Strategy type ("whole_file" or "streaming")
        model_manager: Optional model manager
        audio_processor: Optional audio processor
        voices_dir: Voice directory path
        
    Returns:
        Pipeline instance
        
    Raises:
        ValueError: If strategy is invalid
    """
    from .streaming import StreamingStrategy
    from .whole_file import WholeFileStrategy

    strategies = {
        "whole_file": WholeFileStrategy,
        "streaming": StreamingStrategy
    }

    if strategy not in strategies:
        raise ValueError(
            f"Invalid strategy: {strategy}. "
            f"Must be one of: {', '.join(strategies.keys())}"
        )

    return StrategyPipeline(
        strategies[strategy](model_manager, audio_processor),
        model_manager,
        audio_processor,
        voices_dir
    )