"""Pipeline factory module."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from ..core.config import settings
from ..inference import ModelConfig
from ..structures.audio_schemas import AudioConfig
from ..plugins.hooks import get_plugin_manager


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    model: ModelConfig = ModelConfig()
    audio: AudioConfig = AudioConfig()
    voices_dir: str = "voices"


class PipelineFactory:
    """Factory for creating TTS pipelines."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize factory.
        
        Args:
            config: Optional pipeline configuration
        """
        self._config = config or PipelineConfig()
        self._plugin_manager = None

    @classmethod
    async def create(cls, config: Optional[PipelineConfig] = None) -> "PipelineFactory":
        """Create a new factory instance.
        
        Args:
            config: Optional pipeline configuration
            
        Returns:
            Initialized factory instance
            
        Raises:
            RuntimeError: If initialization fails
        """
        factory = cls(config)
        try:
            factory._plugin_manager = get_plugin_manager()
            logger.info("Pipeline factory initialized")
            return factory
        except Exception as e:
            logger.error(f"Failed to initialize pipeline factory: {e}")
            raise RuntimeError(f"Factory initialization failed: {e}")

    async def create_pipeline(self, pipeline_type: str) -> "Pipeline":
        """Create a pipeline instance.
        
        Args:
            pipeline_type: Type of pipeline ("streaming" or "whole_file")
            
        Returns:
            Pipeline instance
            
        Raises:
            ValueError: If pipeline type is invalid
        """
        try:
            from . import create_pipeline
            return create_pipeline(
                strategy=pipeline_type,
                plugin_manager=self._plugin_manager,
                voices_dir=self._config.voices_dir
            )
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise


# Module-level instance
_factory: Optional[PipelineFactory] = None


async def get_factory(config: Optional[PipelineConfig] = None) -> PipelineFactory:
    """Get or create global factory instance.
    
    Args:
        config: Optional pipeline configuration
        
    Returns:
        PipelineFactory instance
    """
    global _factory
    if _factory is None:
        _factory = await PipelineFactory.create(config)
    return _factory