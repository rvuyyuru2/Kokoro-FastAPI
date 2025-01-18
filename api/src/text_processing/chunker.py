"""Text chunking for TTS preprocessing."""

import re
from typing import Iterator, Optional, Pattern

from loguru import logger

from ..plugins import hookimpl


class TextChunker:
    """Splits text into processable chunks."""

    def __init__(self, max_length: int = 200):
        """Initialize chunker.
        
        Args:
            max_length: Maximum chunk length
            
        Raises:
            ValueError: If max_length is invalid
        """
        if max_length < 1:
            raise ValueError("max_length must be positive")
            
        self.max_length = max_length
        
        # Compile patterns
        self._sentence_end = re.compile(r'[.!?]+')
        self._clause_end = re.compile(r'[,;:]')
        self._whitespace = re.compile(r'\s+')
        
        # Special handling patterns
        self._quote_end = re.compile(r'["""]')
        self._parenthesis_end = re.compile(r'[)}\]]')
        self._number_split = re.compile(r'(?<=\d)(?=[,.]\d)')

    @hookimpl
    def pre_process_text(self, text: str) -> str:
        """Plugin hook for text pre-processing.
        
        Args:
            text: Raw input text
            
        Returns:
            Pre-processed text
        """
        return text

    @hookimpl
    def post_process_text(self, text: str) -> str:
        """Plugin hook for text post-processing.
        
        Args:
            text: Processed chunk
            
        Returns:
            Post-processed text
        """
        return text

    def split_text(self, text: str) -> Iterator[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Yields:
            Text chunks
            
        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValueError("Input must be string")
            
        if not text.strip():
            raise ValueError("Empty input text")

        try:
            # Apply pre-processing hook
            text = self.pre_process_text(text)
            
            # Clean whitespace
            text = self._whitespace.sub(' ', text).strip()
            
            # Process text
            current_pos = 0
            text_length = len(text)
            
            while current_pos < text_length:
                # Get next chunk
                chunk_end = self._find_chunk_end(
                    text,
                    current_pos,
                    min(current_pos + self.max_length, text_length)
                )
                
                # Extract and process chunk
                chunk = text[current_pos:chunk_end].strip()
                if chunk:
                    # Apply post-processing hook
                    chunk = self.post_process_text(chunk)
                    yield chunk
                    
                # Move position
                current_pos = chunk_end
                
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            raise

    def _find_chunk_end(self, text: str, start: int, max_end: int) -> int:
        """Find appropriate end position for chunk.
        
        Args:
            text: Full text
            start: Start position
            max_end: Maximum end position
            
        Returns:
            End position for chunk
        """
        # Try to split on sentence boundary
        pos = self._find_last_match(self._sentence_end, text, start, max_end)
        if pos > start:
            return pos
            
        # Try to split on quote
        pos = self._find_last_match(self._quote_end, text, start, max_end)
        if pos > start:
            return pos
            
        # Try to split on parenthesis
        pos = self._find_last_match(self._parenthesis_end, text, start, max_end)
        if pos > start:
            return pos
            
        # Try to split on clause boundary
        pos = self._find_last_match(self._clause_end, text, start, max_end)
        if pos > start:
            return pos
            
        # Try to split on number boundary
        pos = self._find_last_match(self._number_split, text, start, max_end)
        if pos > start:
            return pos
            
        # Fall back to word boundary
        pos = max_end
        while pos > start and not text[pos - 1].isspace():
            pos -= 1
        if pos > start:
            return pos
            
        # If no word boundary, force split at max_end
        return max_end

    def _find_last_match(
        self,
        pattern: Pattern,
        text: str,
        start: int,
        end: int
    ) -> int:
        """Find last pattern match in range.
        
        Args:
            pattern: Regex pattern
            text: Text to search
            start: Start position
            end: End position
            
        Returns:
            Position of last match or start position if no match
        """
        matches = list(pattern.finditer(text, start, end))
        if matches:
            return matches[-1].end()
        return start


# Module-level instance for convenience
_chunker: Optional[TextChunker] = None


def get_chunker(max_length: int = 200) -> TextChunker:
    """Get or create global chunker instance.
    
    Args:
        max_length: Maximum chunk length
        
    Returns:
        TextChunker instance
    """
    global _chunker
    if _chunker is None:
        _chunker = TextChunker(max_length)
    return _chunker


def split_text(text: str, max_length: int = 200) -> Iterator[str]:
    """Split text using global chunker instance.
    
    Args:
        text: Text to split
        max_length: Maximum chunk length
        
    Returns:
        Iterator of text chunks
    """
    return get_chunker(max_length).split_text(text)