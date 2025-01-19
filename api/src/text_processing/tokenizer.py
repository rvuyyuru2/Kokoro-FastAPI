"""Token generation for TTS preprocessing."""

from typing import List, Dict
from loguru import logger
from ..plugins import hookimpl
from .vocabulary import VOCAB, PAD

@hookimpl
def pre_process_tokens(tokens: List[int]) -> List[int]:
    """Plugin hook for token pre-processing."""
    return tokens

@hookimpl
def post_process_tokens(tokens: List[int]) -> List[int]:
    """Plugin hook for token post-processing."""
    return tokens

def tokenize(phonemes: str) -> List[int]:
    """Convert phonemes string to token IDs exactly as in reference.

    Args:
        phonemes: String of phonemes to tokenize

    Returns:
        List of token IDs
    """
    logger.debug(f"Input to tokenize: '{phonemes}'")
    
    # Apply pre-processing hook
    phonemes = pre_process_tokens(phonemes)
    logger.debug(f"After pre-process: '{phonemes}'")

    # Core tokenization - match reference exactly
    tokens = [i for i in map(VOCAB.get, phonemes) if i is not None]
    
    # Log results
    filtered = ''.join(p for p in phonemes if p in VOCAB)
    logger.debug(f"Final token sequence from: '{filtered}'")
    logger.debug(f"Token IDs: {tokens}")
    
    # Log any filtered characters
    invalid_chars = set(p for p in phonemes if p not in VOCAB)
    if invalid_chars:
        logger.warning(f"Filtered characters: {invalid_chars}")

    # Apply post-processing hook
    tokens = post_process_tokens(tokens)
    
    # Enforce token limit exactly as reference
    if len(tokens) > 510:
        logger.warning('Truncating to 510 tokens')
        tokens = tokens[:510]

    return tokens

def decode_tokens(tokens: List[int]) -> str:
    """Convert token IDs back to phonemes string exactly as in reference.

    Args:
        tokens: List of token IDs

    Returns:
        String of phonemes
    """
    # Create reverse mapping
    id_to_symbol = {i: s for s, i in VOCAB.items()}
    return "".join(id_to_symbol[t] for t in tokens)

def pad_sequence(tokens: List[int], max_length: int) -> List[int]:
    """Pad sequence to fixed length.

    Args:
        tokens: Token sequence to pad
        max_length: Target length

    Returns:
        Padded sequence
    """
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens + [VOCAB[PAD]] * (max_length - len(tokens))