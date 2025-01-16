from typing import Dict

from .strategy_interface import PhonemizationStrategy
from .strategy_factory import PhonemizationStrategyFactory
from .espeak import EspeakStrategy

# Create and configure the default factory instance
factory = PhonemizationStrategyFactory()
factory.register_strategy("espeak", EspeakStrategy, make_default=True)

# Language code mapping for backward compatibility
_LANG_MAP: Dict[str, str] = {
    "a": "en-us",
    "b": "en-gb"
}

def phonemize(text: str, language: str = "a", normalize: bool = True) -> str:
    """Convert text to phonemes (backward compatibility wrapper)
    
    Args:
        text: Text to convert to phonemes
        language: Language code ('a' for US English, 'b' for British English)
        normalize: Whether to normalize text before phonemization
        
    Returns:
        Phonemized text
    """
    if language not in _LANG_MAP:
        raise ValueError(f"Unsupported language code: {language}")
        
    strategy = factory.get_strategy(
        "espeak",
        language=_LANG_MAP[language],
        normalize_text=normalize
    )
    return strategy.phonemize(text)

__all__ = [
    'PhonemizationStrategy',
    'PhonemizationStrategyFactory',
    'EspeakStrategy',
    'factory',
    'phonemize'  # For backward compatibility
]