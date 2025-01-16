from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch


class StyleStrategy(ABC):
    """Base interface for style generation strategies"""
    
    @abstractmethod
    async def generate_style(self, **kwargs) -> torch.Tensor:
        """Generate a voice style based on the strategy's implementation
        
        Args:
            **kwargs: Strategy-specific parameters
            
        Returns:
            torch.Tensor: The generated voice style tensor
        """
        pass
    
    @abstractmethod
    def get_required_params(self) -> Dict[str, Any]:
        """Get the required parameters for this strategy
        
        Returns:
            Dict[str, Any]: Dictionary of parameter names and their types
        """
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate the provided parameters for this strategy
        
        Args:
            params: Dictionary of parameter names and values
            
        Returns:
            bool: True if parameters are valid, False otherwise
            
        Raises:
            ValueError: If parameters are invalid with description
        """
        pass