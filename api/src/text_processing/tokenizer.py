"""Token generation for TTS preprocessing."""

from typing import Dict, List, Optional

from loguru import logger

from .vocabulary import get_vocab, get_vocab_size, PAD


class Tokenizer:
    """Converts phonemes to token IDs."""

    def __init__(self):
        """Initialize tokenizer with vocabulary."""
        self._vocab: Dict[str, int] = {}
        self._pad_id: int = 0
        self._vocab_size: int = 0
        self._load_vocab()

    def _load_vocab(self) -> None:
        """Load vocabulary mapping."""
        try:
            self._vocab = get_vocab()
            self._pad_id = self._vocab[PAD]
            self._vocab_size = get_vocab_size()
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            raise RuntimeError("Tokenizer initialization failed")

    def tokenize(self, phonemes: str) -> List[int]:
        """Convert phoneme string to token IDs.
        
        Args:
            phonemes: String of phonemes to tokenize
            
        Returns:
            List of token IDs
            
        Raises:
            ValueError: If phonemes contain unknown symbols
        """
        if not phonemes:
            raise ValueError("Empty phoneme string")

        try:
            # Split into individual phonemes
            symbols = list(phonemes.replace(" ", "_"))
            
            # Convert to token IDs
            tokens = []
            for symbol in symbols:
                if symbol not in self._vocab:
                    raise ValueError(f"Unknown phoneme symbol: {symbol}")
                tokens.append(self._vocab[symbol])
                
            return tokens
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to phoneme string.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Phoneme string
            
        Raises:
            ValueError: If tokens are invalid
        """
        if not tokens:
            raise ValueError("Empty token list")

        try:
            # Create reverse mapping
            id_to_symbol = {v: k for k, v in self._vocab.items()}
            
            # Convert back to symbols
            symbols = []
            for token in tokens:
                if token not in id_to_symbol:
                    raise ValueError(f"Invalid token ID: {token}")
                symbols.append(id_to_symbol[token])
                
            # Join and clean up
            text = "".join(symbols)
            text = text.replace("_", " ")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Token decoding failed: {e}")
            raise

    def pad_sequence(
        self,
        tokens: List[int],
        max_length: int,
        pad_end: bool = True
    ) -> List[int]:
        """Pad token sequence to fixed length.
        
        Args:
            tokens: Token sequence to pad
            max_length: Target sequence length
            pad_end: Whether to pad at end (True) or start (False)
            
        Returns:
            Padded token sequence
            
        Raises:
            ValueError: If max_length is too small
        """
        if max_length < len(tokens):
            raise ValueError(
                f"Max length {max_length} is smaller than sequence length {len(tokens)}"
            )

        padding = [self._pad_id] * (max_length - len(tokens))
        
        if pad_end:
            return tokens + padding
        else:
            return padding + tokens

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        """Get padding token ID.
        
        Returns:
            ID of padding token
        """
        return self._pad_id


# Module-level instance for convenience
_tokenizer: Optional[Tokenizer] = None


def get_tokenizer() -> Tokenizer:
    """Get or create global tokenizer instance.
    
    Returns:
        Tokenizer instance
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    return _tokenizer


def tokenize(phonemes: str) -> List[int]:
    """Tokenize phonemes using global tokenizer instance.
    
    Args:
        phonemes: Phoneme string to tokenize
        
    Returns:
        List of token IDs
    """
    return get_tokenizer().tokenize(phonemes)


def decode_tokens(tokens: List[int]) -> str:
    """Decode tokens using global tokenizer instance.
    
    Args:
        tokens: Token IDs to decode
        
    Returns:
        Phoneme string
    """
    return get_tokenizer().decode(tokens)


def pad_sequence(
    tokens: List[int],
    max_length: int,
    pad_end: bool = True
) -> List[int]:
    """Pad sequence using global tokenizer instance.
    
    Args:
        tokens: Token sequence to pad
        max_length: Target length
        pad_end: Whether to pad at end
        
    Returns:
        Padded sequence
    """
    return get_tokenizer().pad_sequence(tokens, max_length, pad_end)