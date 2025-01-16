import re
from typing import Generator, Dict, Any

from ....core.config import settings
from ..strategy_interface import ChunkingStrategy


class StaticChunkStrategy(ChunkingStrategy):
    """Strategy for splitting text into chunks using static rules"""
    
    def __init__(self):
        self._config = {
            'max_chunk_size': settings.max_chunk_size
        }
    
    @property
    def name(self) -> str:
        return "static"
        
    def split_text(self, text: str) -> Generator[str, None, None]:
        """Split text into chunks on natural pause points
        
        Args:
            text: Text to split into chunks
            
        Yields:
            Text chunks according to static rules
        """
        max_chunk = self._config['max_chunk_size']
        
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        text = text.strip()
        if not text:
            return
            
        # First split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # For medium-length sentences, split on punctuation
            if len(sentence) > max_chunk:  # Lower threshold for more consistent sizes
                # First try splitting on semicolons and colons
                parts = re.split(r"(?<=[;:])\s+", sentence)
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                        
                    # If part is still long, split on commas
                    if len(part) > max_chunk:
                        subparts = re.split(r"(?<=,)\s+", part)
                        for subpart in subparts:
                            subpart = subpart.strip()
                            if subpart:
                                yield subpart
                    else:
                        yield part
            else:
                yield sentence
                
    def get_config(self) -> Dict[str, Any]:
        """Get the strategy's configuration
        
        Returns:
            Dictionary containing max_chunk_size setting
        """
        return self._config.copy()
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters
        
        Args:
            config: Dictionary containing max_chunk_size
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If max_chunk_size is invalid
        """
        if 'max_chunk_size' in config:
            max_chunk = config['max_chunk_size']
            if not isinstance(max_chunk, int) or max_chunk <= 0:
                raise ValueError("max_chunk_size must be a positive integer")
                
        return True