"""Text processing module for TTS preprocessing."""

from typing import List

from .chunker import split_text
from .normalizer import normalize_text
from .phonemizer import phonemize
from .tokenizer import tokenize


async def process_text(text: str, language: str = "a", max_chunk_length: int = 200) -> List[List[int]]:
    """Process text through complete pipeline asynchronously.

    Args:
        text: Input text
        language: Language code ('a' for US English, 'b' for British English)
        max_chunk_length: Maximum chunk length

    Returns:
        List of token sequences for each chunk
    """
    # Split text into chunks first to enable parallel processing
    chunks = list(split_text(text, max_chunk_length))
    token_sequences = []
    
    # Process each chunk
    for chunk in chunks:
        # Process text through pipeline stages
        # Note: phonemize and tokenize are CPU-bound operations
        # We keep them in the same task to avoid overhead of task switching
        # for small chunks of text
        ps = phonemize(chunk, language)
        tokens = tokenize(ps)
        if tokens:  # Only add if we got valid tokens
            token_sequences.append(tokens)

    return token_sequences