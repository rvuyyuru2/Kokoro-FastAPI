"""Text processing module for TTS preprocessing."""

from .chunker import TextChunker, get_chunker, split_text
from .normalizer import TextNormalizer, get_normalizer, normalize_text
from .phonemizer import (
    Phonemizer,
    PhonemizerBackend,
    EspeakBackend,
    get_phonemizer,
    phonemize
)
from .tokenizer import (
    Tokenizer,
    get_tokenizer,
    tokenize,
    decode_tokens,
    pad_sequence
)
from .vocabulary import (
    get_vocab,
    get_symbols,
    get_vocab_size,
    PAD,
    VOCAB,
    SYMBOLS
)

__all__ = [
    # Chunker
    "TextChunker",
    "get_chunker",
    "split_text",
    
    # Normalizer
    "TextNormalizer",
    "get_normalizer",
    "normalize_text",
    
    # Phonemizer
    "Phonemizer",
    "PhonemizerBackend",
    "EspeakBackend",
    "get_phonemizer",
    "phonemize",
    
    # Tokenizer
    "Tokenizer",
    "get_tokenizer",
    "tokenize",
    "decode_tokens",
    "pad_sequence",
    
    # Vocabulary
    "get_vocab",
    "get_symbols",
    "get_vocab_size",
    "PAD",
    "VOCAB",
    "SYMBOLS"
]

# Version info
__version__ = "1.0.0"
__author__ = "Kokoro TTS Team"
__description__ = "Text processing module for Kokoro TTS"

# Initialize default instances
_default_normalizer = None
_default_phonemizer = None
_default_tokenizer = None
_default_chunker = None


def get_processor(
    language: str = "a",
    max_chunk_length: int = 200
) -> tuple[TextNormalizer, Phonemizer, Tokenizer, TextChunker]:
    """Get complete text processing pipeline.
    
    Args:
        language: Language code ('a' for US English, 'b' for British English)
        max_chunk_length: Maximum chunk length for text splitting
        
    Returns:
        Tuple of:
            - Text normalizer
            - Phonemizer
            - Tokenizer
            - Text chunker
    """
    global _default_normalizer, _default_phonemizer, _default_tokenizer, _default_chunker
    
    # Initialize components if needed
    if _default_normalizer is None:
        _default_normalizer = get_normalizer()
    if _default_phonemizer is None:
        _default_phonemizer = get_phonemizer(language)
    if _default_tokenizer is None:
        _default_tokenizer = get_tokenizer()
    if _default_chunker is None:
        _default_chunker = get_chunker(max_chunk_length)
        
    return (
        _default_normalizer,
        _default_phonemizer,
        _default_tokenizer,
        _default_chunker
    )


def process_text(
    text: str,
    language: str = "a",
    max_chunk_length: int = 200
) -> list[list[int]]:
    """Process text through complete pipeline.
    
    Args:
        text: Input text
        language: Language code
        max_chunk_length: Maximum chunk length
        
    Returns:
        List of token sequences for each chunk
        
    Example:
        >>> tokens = process_text("Hello, world!")
        >>> for chunk_tokens in tokens:
        ...     audio = generate_audio(chunk_tokens)
    """
    # Get processor components
    normalizer, phonemizer, tokenizer, chunker = get_processor(
        language,
        max_chunk_length
    )
    
    # Process text
    normalized = normalizer.normalize(text)
    chunks = chunker.split_text(normalized)
    
    # Process each chunk
    token_sequences = []
    for chunk in chunks:
        phonemes = phonemizer.phonemize(chunk)
        tokens = tokenizer.tokenize(phonemes)
        token_sequences.append(tokens)
        
    return token_sequences