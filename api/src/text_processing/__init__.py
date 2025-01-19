"""Text processing module for TTS preprocessing."""

from typing import List

from .chunker import split_text
from .normalizer import normalize_text
from .phonemizer import phonemize
from .tokenizer import tokenize


def process_text(text: str, language: str = "a", max_chunk_length: int = 200) -> List[List[int]]:
    """Process text through complete pipeline.

    Args:
        text: Input text
        language: Language code ('a' for US English, 'b' for British English)
        max_chunk_length: Maximum chunk length

    Returns:
        List of token sequences for each chunk
    """
    # Match reference implementation flow exactly
    chunks = list(split_text(text, max_chunk_length))
    token_sequences = []
    
    for chunk in chunks:
        # Follow kokoro.py flow exactly
        ps = phonemize(chunk, language)
        tokens = tokenize(ps)
        if tokens:  # Only add if we got valid tokens
            token_sequences.append(tokens)

    return token_sequences