from abc import ABC, abstractmethod
from typing import Dict, Any


class PhonemizationStrategy(ABC):
    """Base interface for phonemization strategies"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass
        
    @abstractmethod
    def phonemize(self, text: str) -> str:
        """Convert text to phonemes
        
        Args:
            text: Text to convert to phonemes
            
        Returns:
            Phonemized text
        """
        pass
        
    @abstractmethod
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages
        
        Returns:
            Dictionary mapping language codes to their descriptions
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