import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import phonemizer
from loguru import logger

from .normalizer import normalize_text

phonemizers = {}


@dataclass
class PhonemeAlignment:
    """Alignment between text and phonemes"""
    text: str
    phonemes: str
    start_idx: int
    end_idx: int


class PhonemizerBackend(ABC):
    """Abstract base class for phonemization backends"""

    @abstractmethod
    def phonemize(self, text: str) -> Tuple[str, List[PhonemeAlignment]]:
        """Convert text to phonemes

        Args:
            text: Text to convert to phonemes

        Returns:
            Tuple of (phonemized text, list of alignments)
        """
        pass


class EspeakBackend(PhonemizerBackend):
    """Espeak-based phonemizer implementation"""

    def __init__(self, language: str):
        """Initialize espeak backend

        Args:
            language: Language code ('en-us' or 'en-gb')
        """
        self.backend = phonemizer.backend.EspeakBackend(
            language=language, preserve_punctuation=True, with_stress=True
        )
        self.language = language

    def phonemize(self, text: str) -> Tuple[str, List[PhonemeAlignment]]:
        """Convert text to phonemes using espeak

        Args:
            text: Text to convert to phonemes

        Returns:
            Tuple of (phonemized text, list of alignments)
        """
        # Split text into words while preserving spaces and punctuation
        words = re.findall(r'\S+|\s+', text)
        alignments = []
        phonemes_list = []
        
        # Track current position in text and phonemes
        text_pos = 0
        
        # Process each word
        for word in words:
            # Skip pure whitespace/punctuation
            if not re.search(r'[a-zA-Z]', word):
                text_pos += len(word)
                continue
                
            # Phonemize single word
            ps = self.backend.phonemize([word])
            word_phonemes = ps[0] if ps else ""
            
            # Handle special cases
            word_phonemes = word_phonemes.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ").replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ")
            word_phonemes = word_phonemes.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k").replace("ɬ", "l")
            word_phonemes = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", word_phonemes)
            word_phonemes = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', "z", word_phonemes)

            # Language-specific rules
            if self.language == "en-us":
                word_phonemes = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", word_phonemes)
            
            # Create alignment
            if word_phonemes:
                alignment = PhonemeAlignment(
                    text=word,
                    phonemes=word_phonemes,
                    start_idx=text_pos,
                    end_idx=text_pos + len(word)
                )
                logger.debug(f"Created alignment: text='{word}' phonemes='{word_phonemes}' start={text_pos} end={text_pos + len(word)}")
                alignments.append(alignment)
                phonemes_list.append(word_phonemes)
            else:
                logger.debug(f"No phonemes generated for word: '{word}'")
            
            text_pos += len(word)

        # Combine all phonemes
        full_phonemes = " ".join(phonemes_list)
        
        return full_phonemes.strip(), alignments


def create_phonemizer(language: str = "a") -> PhonemizerBackend:
    """Factory function to create phonemizer backend

    Args:
        language: Language code ('a' for US English, 'b' for British English)

    Returns:
        Phonemizer backend instance
    """
    # Map language codes to espeak language codes
    lang_map = {"a": "en-us", "b": "en-gb"}

    if language not in lang_map:
        raise ValueError(f"Unsupported language code: {language}")

    return EspeakBackend(lang_map[language])


def phonemize(text: str, language: str = "a", normalize: bool = True) -> Tuple[str, List[PhonemeAlignment]]:
    """Convert text to phonemes

    Args:
        text: Text to convert to phonemes
        language: Language code ('a' for US English, 'b' for British English)
        normalize: Whether to normalize text before phonemization

    Returns:
        Tuple of (phonemized text, list of alignments)
    """
    global phonemizers
    
    try:
        if normalize:
            text = normalize_text(text)
            
        if language not in phonemizers:
            phonemizers[language] = create_phonemizer(language)
            
        return phonemizers[language].phonemize(text)
        
    except Exception as e:
        logger.error(f"Phonemization failed: {e}")
        return "", []