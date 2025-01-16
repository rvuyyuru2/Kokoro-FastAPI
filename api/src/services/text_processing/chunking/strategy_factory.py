from typing import Dict, Type

from .strategy_interface import ChunkingStrategy


class ChunkingStrategyFactory:
    """Factory for creating text chunking strategies"""
    
    def __init__(self):
        self._strategies: Dict[str, Type[ChunkingStrategy]] = {}
        
    def register_strategy(self, name: str, strategy_class: Type[ChunkingStrategy]):
        """Register a new chunking strategy
        
        Args:
            name: Name to register the strategy under
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, ChunkingStrategy):
            raise ValueError(
                f"Strategy class must implement ChunkingStrategy interface: {strategy_class}"
            )
        self._strategies[name] = strategy_class
        
    def get_strategy(self, name: str, **config) -> ChunkingStrategy:
        """Get a chunking strategy by name
        
        Args:
            name: Name of the strategy to get
            **config: Strategy-specific configuration parameters
            
        Returns:
            ChunkingStrategy: Instance of the requested strategy
            
        Raises:
            ValueError: If strategy name not found
        """
        strategy_class = self._strategies.get(name)
        if not strategy_class:
            available = ", ".join(self._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{name}'. Available strategies: {available}"
            )
            
        strategy = strategy_class()
        if config:
            strategy.validate_config(config)
            strategy.get_config().update(config)
            
        return strategy
        
    def list_strategies(self) -> Dict[str, Type[ChunkingStrategy]]:
        """Get all registered strategies
        
        Returns:
            Dict mapping strategy names to their classes
        """
        return self._strategies.copy()