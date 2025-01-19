"""TTS pipeline module."""

from typing import AsyncIterator, Optional, Union, Any

from .base import Pipeline
from .factory import PipelineFactory, get_factory, PipelineConfig
from .streaming import StreamingPipeline
from .whole_file import WholeFilePipeline
from ..inference import ModelManager
from ..audio_processing import AudioProcessor


def create_pipeline(
    strategy: str = "whole_file",
    model_manager: Optional[ModelManager] = None,
    audio_processor: Optional[AudioProcessor] = None,
    voices_dir: str = "voices",
    plugin_manager: Optional[Any] = None
) -> Pipeline:
    """Create pipeline instance.
    
    Args:
        strategy: Pipeline type ("whole_file" or "streaming")
        model_manager: Optional model manager
        audio_processor: Optional audio processor
        voices_dir: Voice directory path
        plugin_manager: Optional plugin manager
        
    Returns:
        Pipeline instance
        
    Raises:
        ValueError: If strategy is invalid
    """
    pipelines = {
        "whole_file": WholeFilePipeline,
        "streaming": StreamingPipeline
    }

    if strategy not in pipelines:
        raise ValueError(
            f"Invalid strategy: {strategy}. "
            f"Must be one of: {', '.join(pipelines.keys())}"
        )

    return pipelines[strategy](
        model_manager=model_manager,
        audio_processor=audio_processor,
        voices_dir=voices_dir,
        plugin_manager=plugin_manager
    )


__all__ = [
    # Core types
    "Pipeline",
    "PipelineConfig",
    
    # Pipeline implementations
    "StreamingPipeline",
    "WholeFilePipeline",
    
    # Factory
    "PipelineFactory",
    "get_factory",
    "create_pipeline"
]