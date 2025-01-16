from abc import ABC, abstractmethod
from typing import Dict, Any, List


class TokenizationStrategy(ABC):
    """Base interface for tokenization strategies"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass
        
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        pass
        
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        pass
        
    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        pass
        
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary
        
        Returns:
            Number of tokens in vocabulary
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
        
    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure the strategy with new settings
        
        Args:
            **kwargs: Strategy-specific configuration parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass