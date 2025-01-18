"""Phoneme conversion for TTS preprocessing."""

import re
from abc import ABC, abstractmethod
from typing import Dict, Optional

import phonemizer
from loguru import logger

from ..plugins import hookimpl
from .normalizer import normalize_text


class PhonemizerBackend(ABC):
    """Abstract base class for phonemization backends."""

    @abstractmethod
    def phonemize(self, text: str) -> str:
        """Convert text to phonemes.
        
        Args:
            text: Text to convert to phonemes
            
        Returns:
            Phonemized text
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If phonemization fails
        """
        pass


class EspeakBackend(PhonemizerBackend):
    """Espeak-based phonemizer implementation."""

    def __init__(self, language: str):
        """Initialize espeak backend.
        
        Args:
            language: Language code ('en-us' or 'en-gb')
            
        Raises:
            RuntimeError: If backend initialization fails
        """
        try:
            self.backend = phonemizer.backend.EspeakBackend(
                language=language,
                preserve_punctuation=True,
                with_stress=True
            )
            self.language = language
        except Exception as e:
            raise RuntimeError(f"Failed to initialize espeak backend: {e}")

    def phonemize(self, text: str) -> str:
        """Convert text to phonemes using espeak.
        
        Args:
            text: Text to convert to phonemes
            
        Returns:
            Phonemized text
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If phonemization fails
        """
        if not text:
            raise ValueError("Empty input text")

        try:
            # Phonemize text
            ps = self.backend.phonemize([text])
            ps = ps[0] if ps else ""

            # Apply phoneme rules
            ps = self._apply_rules(ps)
            
            return ps.strip()
            
        except Exception as e:
            raise RuntimeError(f"Phonemization failed: {e}")

    def _apply_rules(self, text: str) -> str:
        """Apply language-specific phoneme rules.
        
        Args:
            text: Phonemized text
            
        Returns:
            Text with rules applied
        """
        # Common replacements
        text = text.replace("ʲ", "j")
        text = text.replace("r", "ɹ")
        text = text.replace("x", "k")
        text = text.replace("ɬ", "l")

        # Fix specific word patterns
        text = text.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ")
        text = text.replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ")

        # Add spaces in specific contexts
        text = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", text)
        text = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', "z", text)

        # Language-specific rules
        if self.language == "en-us":
            text = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", text)

        return text


class Phonemizer:
    """Main phonemizer class with plugin support."""

    def __init__(self, language: str = "a"):
        """Initialize phonemizer.
        
        Args:
            language: Language code ('a' for US English, 'b' for British English)
            
        Raises:
            ValueError: If language code is invalid
        """
        # Map language codes to espeak language codes
        self._lang_map = {
            "a": "en-us",
            "b": "en-gb"
        }
        
        if language not in self._lang_map:
            raise ValueError(f"Unsupported language code: {language}")
            
        self.backend = EspeakBackend(self._lang_map[language])

    @hookimpl
    def pre_process_phonemes(self, phonemes: str) -> str:
        """Plugin hook for phoneme pre-processing.
        
        Args:
            phonemes: Raw phoneme string
            
        Returns:
            Pre-processed phonemes
        """
        return phonemes

    @hookimpl
    def post_process_phonemes(self, phonemes: str) -> str:
        """Plugin hook for phoneme post-processing.
        
        Args:
            phonemes: Processed phoneme string
            
        Returns:
            Post-processed phonemes
        """
        return phonemes

    def phonemize(self, text: str, normalize: bool = True) -> str:
        """Convert text to phonemes.
        
        Args:
            text: Text to convert to phonemes
            normalize: Whether to normalize text before phonemization
            
        Returns:
            Phonemized text
            
        Raises:
            ValueError: If text is invalid
            RuntimeError: If phonemization fails
        """
        if normalize:
            text = normalize_text(text)

        # Apply pre-processing hook
        text = self.pre_process_phonemes(text)
            
        # Convert to phonemes
        phonemes = self.backend.phonemize(text)
        
        # Apply post-processing hook
        phonemes = self.post_process_phonemes(phonemes)
            
        return phonemes


# Module-level instance for convenience
_phonemizer: Optional[Phonemizer] = None


def get_phonemizer(language: str = "a") -> Phonemizer:
    """Get or create global phonemizer instance.
    
    Args:
        language: Language code
        
    Returns:
        Phonemizer instance
    """
    global _phonemizer
    if _phonemizer is None:
        _phonemizer = Phonemizer(language)
    return _phonemizer


def phonemize(text: str, language: str = "a", normalize: bool = True) -> str:
    """Phonemize text using global phonemizer instance.
    
    Args:
        text: Text to convert to phonemes
        language: Language code
        normalize: Whether to normalize text
        
    Returns:
        Phonemized text
    """
    return get_phonemizer(language).phonemize(text, normalize)