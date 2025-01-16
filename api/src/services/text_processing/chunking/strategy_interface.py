from abc import ABC, abstractmethod
from typing import Generator, Dict, Any


class ChunkingStrategy(ABC):
    """Base interface for text chunking strategies"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass
        
    @abstractmethod
    def split_text(self, text: str) -> Generator[str, None, None]:
        """Split text into chunks using the strategy's approach
        
        Args:
            text: Input text to split into chunks
            
        Yields:
            Text chunks according to the strategy
        """
        pass
        
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the strategy's configuration
        
        Returns:
            Dictionary containing strategy-specific settings
        """
        pass
        
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass