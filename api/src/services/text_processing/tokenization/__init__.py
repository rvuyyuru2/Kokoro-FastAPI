from typing import List

from .strategy_interface import TokenizationStrategy
from .strategy_factory import TokenizationStrategyFactory
from .basic import BasicTokenizationStrategy

# Create and configure the default factory instance
factory = TokenizationStrategyFactory()
factory.register_strategy("basic", BasicTokenizationStrategy, make_default=True)

# Get default strategy instance for backward compatibility
_default_strategy = factory.get_strategy("basic")

# Expose vocabulary for backward compatibility
VOCAB = _default_strategy.get_vocab()

def tokenize(text: str) -> List[int]:
    """Convert text to token IDs (backward compatibility wrapper)
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of token IDs
    """
    return _default_strategy.tokenize(text)
    
def decode_tokens(tokens: List[int]) -> str:
    """Convert token IDs back to text (backward compatibility wrapper)
    
    Args:
        tokens: List of token IDs
        
    Returns:
        Decoded text
    """
    return _default_strategy.decode(tokens)

__all__ = [
    'TokenizationStrategy',
    'TokenizationStrategyFactory',
    'BasicTokenizationStrategy',
    'factory',
    'VOCAB',           # For backward compatibility
    'tokenize',        # For backward compatibility
    'decode_tokens'    # For backward compatibility
]