"""Phoneme conversion for TTS preprocessing."""

import re
import phonemizer
from loguru import logger
from ..plugins import hookimpl
from .normalizer import normalize_text
from .vocabulary import VOCAB

# Initialize phonemizers exactly as in reference
logger.debug("Initializing phonemizers")
phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
    b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True),
)
logger.debug("Phonemizers initialized")

@hookimpl
def pre_process_phonemes(phonemes: str) -> str:
    """Plugin hook for phoneme pre-processing."""
    return phonemes

@hookimpl
def post_process_phonemes(phonemes: str) -> str:
    """Plugin hook for phoneme post-processing."""
    return phonemes

def phonemize(text: str, language: str = "a", normalize: bool = True) -> str:
    """Convert text to phonemes exactly as in reference implementation.

    Args:
        text: Text to convert to phonemes
        language: Language code ('a' for US English, 'b' for British English)
        normalize: Whether to normalize text before phonemization

    Returns:
        Phonemized text
    """
    # logger.debug(f"Input text to phonemize: '{text}'")
    
    # Apply pre-processing hook
    text = pre_process_phonemes(text)
    # logger.debug(f"After pre-process phonemes: '{text}'")

    # Normalize if requested
    if normalize:
        text = normalize_text(text)
        # logger.debug(f"After normalization: '{text}'")

    # Core phonemization - match reference exactly
    ps = phonemizers[language].phonemize([text])
    ps = ps[0] if ps else ''
    
    # Apply exact same replacements as reference
    ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
    ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
    ps = re.sub(r'(?<=[a-zɹː])(?=hˈʌndɹɪd)', ' ', ps)
    ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', 'z', ps)
    if language == 'a':
        ps = re.sub(r'(?<=nˈaɪn)ti(?!ː)', 'di', ps)
    
    # Filter exactly like reference
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    # logger.debug(f"After phonemization: '{ps}'")

    # Apply post-processing hook
    ps = post_process_phonemes(ps)
    # logger.debug(f"Final phonemes: '{ps}'")
    
    return ps.strip()
