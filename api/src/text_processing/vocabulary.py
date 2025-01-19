"""Vocabulary management for TTS preprocessing."""

from typing import Dict, List
from ..plugins import hookimpl

# Constants from reference implementation
PAD = "$"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

@hookimpl
def pre_process_vocab(symbols: List[str]) -> List[str]:
    """Plugin hook for vocabulary pre-processing."""
    return symbols

@hookimpl
def post_process_vocab(vocab: Dict[str, int]) -> Dict[str, int]:
    """Plugin hook for vocabulary post-processing."""
    return vocab

def get_vocab() -> Dict[str, int]:
    """Get vocabulary mapping from reference implementation.
    
    Returns:
        Dictionary mapping symbols to token IDs
    """
    # Start with base symbols
    symbols = [PAD] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    
    # Apply pre-processing hook
    symbols = pre_process_vocab(symbols)
    
    # Create mapping exactly as in reference
    vocab = {}
    for i in range(len(symbols)):
        vocab[symbols[i]] = i
        
    # Apply post-processing hook
    vocab = post_process_vocab(vocab)
    
    return vocab

# Initialize vocabulary at module level
VOCAB = get_vocab()