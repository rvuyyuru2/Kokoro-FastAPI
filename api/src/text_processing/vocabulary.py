"""Vocabulary management for tokenization."""

from typing import Dict, List

# Special tokens
PAD = "$"

# Character sets
PUNCTUATION = ";:,.!?¡¿—…\"«»"" "
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧ"
    "ʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)


def load_vocabulary() -> Dict[str, int]:
    """Load the vocabulary mapping.
    
    Returns:
        Dictionary mapping symbols to token IDs
    """
    # Combine all symbols
    symbols = [PAD] + list(PUNCTUATION) + list(LETTERS) + list(LETTERS_IPA)
    
    # Create mapping
    return {symbol: i for i, symbol in enumerate(symbols)}


def get_symbol_table() -> Dict[str, str]:
    """Get human-readable symbol descriptions.
    
    Returns:
        Dictionary mapping symbols to descriptions
    """
    return {
        PAD: "Padding token",
        " ": "Space",
        ".": "Period",
        ",": "Comma",
        "!": "Exclamation",
        "?": "Question",
        "-": "Hyphen",
        "'": "Apostrophe",
        '"': "Quote",
        "ˈ": "Primary stress",
        "ˌ": "Secondary stress",
        "ː": "Long sound",
        "ə": "Schwa",
        "æ": "Near-open front unrounded",
        "ɑ": "Open back unrounded",
        "ɔ": "Open-mid back rounded",
        "ɛ": "Open-mid front unrounded",
        "ɪ": "Near-close near-front unrounded",
        "ʊ": "Near-close near-back rounded",
        "ʌ": "Open-mid back unrounded",
        "ŋ": "Velar nasal",
        "ʃ": "Voiceless postalveolar fricative",
        "θ": "Voiceless dental fricative",
        "ð": "Voiced dental fricative",
        "ʒ": "Voiced postalveolar fricative",
        "ʤ": "Voiced postalveolar affricate",
        "ʧ": "Voiceless postalveolar affricate"
    }


# Initialize vocabulary at module level
VOCAB = load_vocabulary()
SYMBOLS = get_symbol_table()


def get_vocab() -> Dict[str, int]:
    """Get the vocabulary dictionary.
    
    Returns:
        Dictionary mapping symbols to token IDs
    """
    return VOCAB


def get_symbols() -> Dict[str, str]:
    """Get the symbol descriptions.
    
    Returns:
        Dictionary mapping symbols to descriptions
    """
    return SYMBOLS


def get_vocab_size() -> int:
    """Get the vocabulary size.
    
    Returns:
        Number of tokens in vocabulary
    """
    return len(VOCAB)