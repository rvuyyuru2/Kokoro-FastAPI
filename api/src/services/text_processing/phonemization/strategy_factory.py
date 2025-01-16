from typing import Dict, Type, Optional

from .strategy_interface import PhonemizationStrategy


class PhonemizationStrategyFactory:
    """Factory for creating phonemization strategies"""
    
    def __init__(self):
        self._strategies: Dict[str, Type[PhonemizationStrategy]] = {}
        self._default_strategy: Optional[str] = None
        
    def register_strategy(
        self, 
        name: str, 
        strategy_class: Type[PhonemizationStrategy],
        make_default: bool = False
    ):
        """Register a new phonemization strategy
        
        Args:
            name: Name to register the strategy under
            strategy_class: Strategy class to register
            make_default: Whether to make this the default strategy
        """
        if not issubclass(strategy_class, PhonemizationStrategy):
            raise ValueError(
                f"Strategy class must implement PhonemizationStrategy interface: {strategy_class}"
            )
        self._strategies[name] = strategy_class
        
        if make_default or self._default_strategy is None:
            self._default_strategy = name
            
    def get_strategy(self, name: Optional[str] = None, **config) -> PhonemizationStrategy:
        """Get a phonemization strategy by name
        
        Args:
            name: Name of the strategy to get, or None for default
            **config: Strategy-specific configuration parameters
            
        Returns:
            PhonemizationStrategy: Instance of the requested strategy
            
        Raises:
            ValueError: If strategy name not found
        """
        if name is None:
            if self._default_strategy is None:
                raise ValueError("No default strategy configured")
            name = self._default_strategy
            
        strategy_class = self._strategies.get(name)
        if not strategy_class:
            available = ", ".join(self._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{name}'. Available strategies: {available}"
            )
            
        strategy = strategy_class()
        if config:
            strategy.validate_config(config)
            strategy.configure(**config)
            
        return strategy
        
    def list_strategies(self) -> Dict[str, Dict[str, str]]:
        """Get all registered strategies and their supported languages
        
        Returns:
            Dict mapping strategy names to their supported languages
        """
        return {
            name: strategy_class().get_supported_languages()
            for name, strategy_class in self._strategies.items()
        }