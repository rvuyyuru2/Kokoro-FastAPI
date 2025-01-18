"""Text normalization for TTS preprocessing."""

import re
from typing import Dict, List, Optional, Pattern

from loguru import logger

from ..plugins import hookimpl


class TextNormalizer:
    """Handles text normalization for TTS input."""

    def __init__(self):
        """Initialize normalizer with patterns."""
        # Common patterns
        self._whitespace = re.compile(r'\s+')
        self._multiple_periods = re.compile(r'\.{2,}')
        self._multiple_exclaim = re.compile(r'!{2,}')
        self._multiple_question = re.compile(r'\?{2,}')
        self._url = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Number patterns
        self._numbers = re.compile(r'\d+')
        self._ordinal = re.compile(r'\d+(st|nd|rd|th)')
        self._time = re.compile(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?')
        self._date = re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}')
        
        # Special character replacements using unicode escapes
        self._replacements = {
            '\u201c': '"',  # left double quote
            '\u201d': '"',  # right double quote
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u2014': '-',  # em dash
            '\u2013': '-',  # en dash
            '\u2026': '...', # ellipsis
            '\u00a0': ' ',  # non-breaking space
            '\u200b': '',   # zero-width space
            '\u200c': '',   # zero-width non-joiner
            '\u200d': '',   # zero-width joiner
        }
        
        # Compile replacement pattern
        self._replacement_pattern = re.compile(
            '|'.join(map(re.escape, self._replacements.keys()))
        )

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
            text: Normalized text
            
        Returns:
            Post-processed text
        """
        return text

    def normalize(self, text: str) -> str:
        """Normalize input text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
            
        Raises:
            ValueError: If input text is invalid
        """
        if not isinstance(text, str):
            raise ValueError("Input must be string")
        
        if not text.strip():
            raise ValueError("Input text is empty")

        try:
            # Apply pre-processing hook
            text = self.pre_process_text(text)
            
            # Basic cleanup
            text = self._clean_whitespace(text)
            text = self._replace_special_chars(text)
            
            # Handle special cases
            text = self._normalize_urls(text)
            text = self._normalize_numbers(text)
            text = self._normalize_punctuation(text)
            
            # Apply post-processing hook
            text = self.post_process_text(text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text normalization failed: {e}")
            raise

    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        return self._whitespace.sub(' ', text)

    def _replace_special_chars(self, text: str) -> str:
        """Replace special characters with normalized versions.
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters replaced
        """
        return self._replacement_pattern.sub(
            lambda m: self._replacements[m.group()],
            text
        )

    def _normalize_urls(self, text: str) -> str:
        """Handle URLs in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized URLs
        """
        return self._url.sub('[URL]', text)

    def _normalize_numbers(self, text: str) -> str:
        """Normalize numbers and dates.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized numbers
        """
        # Handle dates
        text = self._date.sub(
            lambda m: self._format_date(m.group()),
            text
        )
        
        # Handle times
        text = self._time.sub(
            lambda m: self._format_time(m.group()),
            text
        )
        
        # Handle ordinals
        text = self._ordinal.sub(
            lambda m: self._format_ordinal(m.group()),
            text
        )
        
        # Handle regular numbers
        text = self._numbers.sub(
            lambda m: self._format_number(m.group()),
            text
        )
        
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized punctuation
        """
        # Normalize multiple periods
        text = self._multiple_periods.sub('...', text)
        
        # Normalize multiple exclamation marks
        text = self._multiple_exclaim.sub('!', text)
        
        # Normalize multiple question marks
        text = self._multiple_question.sub('?', text)
        
        return text

    def _format_date(self, date: str) -> str:
        """Format date string (placeholder).
        
        Args:
            date: Date string
            
        Returns:
            Formatted date
        """
        # TODO: Implement proper date formatting
        return date

    def _format_time(self, time: str) -> str:
        """Format time string (placeholder).
        
        Args:
            time: Time string
            
        Returns:
            Formatted time
        """
        # TODO: Implement proper time formatting
        return time

    def _format_ordinal(self, number: str) -> str:
        """Format ordinal number (placeholder).
        
        Args:
            number: Ordinal number string
            
        Returns:
            Formatted ordinal
        """
        # TODO: Implement proper ordinal formatting
        return number

    def _format_number(self, number: str) -> str:
        """Format regular number (placeholder).
        
        Args:
            number: Number string
            
        Returns:
            Formatted number
        """
        # TODO: Implement proper number formatting
        return number


# Module-level instance for convenience
_normalizer: Optional[TextNormalizer] = None


def get_normalizer() -> TextNormalizer:
    """Get or create global normalizer instance.
    
    Returns:
        TextNormalizer instance
    """
    global _normalizer
    if _normalizer is None:
        _normalizer = TextNormalizer()
    return _normalizer


def normalize_text(text: str) -> str:
    """Normalize text using global normalizer instance.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    return get_normalizer().normalize(text)