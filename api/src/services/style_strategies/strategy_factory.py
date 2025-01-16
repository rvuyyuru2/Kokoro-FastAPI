from typing import Dict, Type

from .style_interface import StyleStrategy
from .combine import CombineVoicesStrategy


class StyleStrategyFactory:
    """Factory for creating style generation strategies"""
    
    def __init__(self, tts_service):
        self.tts_service = tts_service
        self._strategies: Dict[str, Type[StyleStrategy]] = {
            'combine': CombineVoicesStrategy
        }
        
    def register_strategy(self, name: str, strategy_class: Type[StyleStrategy]):
        """Register a new style generation strategy
        
        Args:
            name: Name to register the strategy under
            strategy_class: The strategy class to register
        """
        if not issubclass(strategy_class, StyleStrategy):
            raise ValueError(
                f"Strategy class must implement StyleStrategy interface: {strategy_class}"
            )
        self._strategies[name] = strategy_class
        
    def get_strategy(self, name: str) -> StyleStrategy:
        """Get a style generation strategy by name
        
        Args:
            name: Name of the strategy to get
            
        Returns:
            StyleStrategy: Instance of the requested strategy
            
        Raises:
            ValueError: If strategy name not found
        """
        if name not in self._strategies:
            raise ValueError(f"Style strategy not found: {name}")
            
        strategy_class = self._strategies[name]
        return strategy_class(self.tts_service)
        
    def list_strategies(self) -> Dict[str, Type[StyleStrategy]]:
        """Get all registered strategies
        
        Returns:
            Dict mapping strategy names to their classes
        """
        return self._strategies.copy()