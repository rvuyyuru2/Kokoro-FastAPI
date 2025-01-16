from typing import Dict, Any, List

from ..strategy_interface import TokenizationStrategy


class BasicTokenizationStrategy(TokenizationStrategy):
    """Basic vocabulary-based tokenization strategy"""
    
    def __init__(self):
        self._config = {
            'pad_token': '$',
            'include_punctuation': True,
            'include_letters': True,
            'include_ipa': True
        }
        self._vocab = None
        self._init_vocab()
        
    @property
    def name(self) -> str:
        return "basic"
        
    def _init_vocab(self):
        """Initialize or reinitialize the vocabulary"""
        symbols = []
        
        # Add pad token
        symbols.append(self._config['pad_token'])
        
        # Add punctuation if configured
        if self._config['include_punctuation']:
            punctuation = ';:,.!?¡¿—…"«»"" '
            symbols.extend(list(punctuation))
            
        # Add letters if configured
        if self._config['include_letters']:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            symbols.extend(list(letters))
            
        # Add IPA symbols if configured
        if self._config['include_ipa']:
            letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
            symbols.extend(list(letters_ipa))
            
        # Create vocabulary mapping
        self._vocab = {symbol: i for i, symbol in enumerate(symbols)}
        self._id_to_symbol = {i: s for s, i in self._vocab.items()}
        
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        return [i for i in map(self._vocab.get, text) if i is not None]
        
    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        return "".join(self._id_to_symbol[t] for t in tokens)
        
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping"""
        return self._vocab.copy()
        
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary"""
        return len(self._vocab)
        
    def get_config(self) -> Dict[str, Any]:
        """Get the strategy's configuration"""
        return self._config.copy()
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'pad_token' in config:
            if not isinstance(config['pad_token'], str) or len(config['pad_token']) != 1:
                raise ValueError("pad_token must be a single character")
                
        for key in ['include_punctuation', 'include_letters', 'include_ipa']:
            if key in config and not isinstance(config[key], bool):
                raise ValueError(f"{key} must be a boolean")
                
        return True
        
    def configure(self, **kwargs) -> None:
        """Configure the strategy with new settings
        
        Args:
            **kwargs: Configuration parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.validate_config(kwargs)
        self._config.update(kwargs)
        self._init_vocab()