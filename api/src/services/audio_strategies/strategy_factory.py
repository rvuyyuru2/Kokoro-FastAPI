from typing import Dict, Type

from .strategy_interface import AudioStrategy
from .streaming import StreamingStrategy
from .default_stitch import DefaultStitchStrategy
from ..service_interfaces import TTSServiceProtocol


class AudioStrategyFactory:
    """Factory for creating audio processing strategies"""
    
    def __init__(self, tts_service: TTSServiceProtocol):
        self.tts_service = tts_service
        self._strategies: Dict[str, Type[AudioStrategy]] = {
            "streaming": StreamingStrategy,
            "default_stitch": DefaultStitchStrategy,
        }
    
    def get_strategy(self, strategy_name: str = "default_stitch") -> AudioStrategy:
        """Get an audio processing strategy by name
        
        Args:
            strategy_name: Name of the strategy to use
            
        Returns:
            AudioStrategy instance
            
        Raises:
            ValueError: If strategy_name is not recognized
        """
        strategy_class = self._strategies.get(strategy_name)
        if not strategy_class:
            available = ", ".join(self._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available strategies: {available}"
            )
        
        return strategy_class(self.tts_service)
    
    def register_strategy(self, name: str, strategy_class: Type[AudioStrategy]):
        """Register a new strategy
        
        Args:
            name: Name to register the strategy under
            strategy_class: Strategy class to register
        """
        self._strategies[name] = strategy_class