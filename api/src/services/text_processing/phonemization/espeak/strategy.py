import re
from typing import Dict, Any

import phonemizer

from ..strategy_interface import PhonemizationStrategy
from ...normalizer import normalize_text


class EspeakStrategy(PhonemizationStrategy):
    """Espeak-based phonemization strategy"""
    
    def __init__(self):
        self._config = {
            'language': 'en-us',
            'preserve_punctuation': True,
            'with_stress': True,
            'normalize_text': True
        }
        self._backend = None
        self._init_backend()
        
    @property
    def name(self) -> str:
        return "espeak"
        
    def _init_backend(self):
        """Initialize or reinitialize the espeak backend"""
        self._backend = phonemizer.backend.EspeakBackend(
            language=self._config['language'],
            preserve_punctuation=self._config['preserve_punctuation'],
            with_stress=self._config['with_stress']
        )
        
    def phonemize(self, text: str) -> str:
        """Convert text to phonemes using espeak
        
        Args:
            text: Text to convert to phonemes
            
        Returns:
            Phonemized text
        """
        if self._config['normalize_text']:
            text = normalize_text(text)
            
        # Phonemize text
        ps = self._backend.phonemize([text])
        ps = ps[0] if ps else ""
        
        # Handle special cases
        ps = ps.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ").replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ")
        ps = ps.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k").replace("ɬ", "l")
        ps = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", ps)
        ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', "z", ps)
        
        # Language-specific rules
        if self._config['language'] == "en-us":
            ps = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", ps)
            
        return ps.strip()
        
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages
        
        Returns:
            Dictionary mapping language codes to their descriptions
        """
        return {
            'a': 'US English (en-us)',
            'b': 'British English (en-gb)'
        }
        
    def get_config(self) -> Dict[str, Any]:
        """Get the strategy's configuration"""
        return self._config.copy()
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'language' in config:
            lang = config['language']
            if lang not in ['en-us', 'en-gb']:
                raise ValueError("language must be 'en-us' or 'en-gb'")
                
        for key in ['preserve_punctuation', 'with_stress', 'normalize_text']:
            if key in config and not isinstance(config[key], bool):
                raise ValueError(f"{key} must be a boolean")
                
        return True
        
    def configure(self, **kwargs) -> None:
        """Configure the strategy with new settings
        
        Args:
            **kwargs: Configuration parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.validate_config(kwargs)
        self._config.update(kwargs)
        
        # Reinitialize backend if relevant settings changed
        backend_settings = {'language', 'preserve_punctuation', 'with_stress'}
        if any(key in kwargs for key in backend_settings):
            self._init_backend()