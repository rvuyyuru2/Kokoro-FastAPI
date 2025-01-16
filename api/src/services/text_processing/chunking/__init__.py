from .strategy_interface import ChunkingStrategy
from .strategy_factory import ChunkingStrategyFactory
from .static import StaticChunkStrategy
from .dynamic import DynamicChunkStrategy

# Create and configure the default factory instance
factory = ChunkingStrategyFactory()
factory.register_strategy("static", StaticChunkStrategy)
factory.register_strategy("dynamic", DynamicChunkStrategy)

# Default strategy for backward compatibility
default_strategy = factory.get_strategy("static")
split_text = default_strategy.split_text

__all__ = [
    'ChunkingStrategy',
    'ChunkingStrategyFactory',
    'StaticChunkStrategy',
    'DynamicChunkStrategy',
    'factory',
    'split_text'  # For backward compatibility
]