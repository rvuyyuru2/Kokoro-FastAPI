"""Tests for text processing module."""

import pytest

from ..text_processing import (
    TextNormalizer,
    Phonemizer,
    Tokenizer,
    TextChunker,
    process_text
)
from ..text_processing.vocabulary import get_vocab, get_symbols


class TestTextNormalizer:
    """Test text normalization."""

    def test_basic_normalization(self):
        """Test basic text cleanup."""
        normalizer = TextNormalizer()
        text = '"Hello,   world!"'
        result = normalizer.normalize(text)
        assert result == 'Hello, world!'

    def test_special_characters(self):
        """Test special character handling."""
        normalizer = TextNormalizer()
        text = "It's—a "test" of…special chars"
        result = normalizer.normalize(text)
        assert "'" in result  # Preserves apostrophes
        assert '"' not in result  # Removes quotes
        assert "..." in result  # Normalizes ellipsis

    def test_whitespace(self):
        """Test whitespace handling."""
        normalizer = TextNormalizer()
        text = "Multiple    spaces\nand\tTabs"
        result = normalizer.normalize(text)
        assert "  " not in result
        assert "\t" not in result
        assert result == "Multiple spaces and Tabs"

    def test_empty_input(self):
        """Test empty input handling."""
        normalizer = TextNormalizer()
        with pytest.raises(ValueError):
            normalizer.normalize("")
        with pytest.raises(ValueError):
            normalizer.normalize("   ")


class TestPhonemizer:
    """Test phoneme conversion."""

    def test_basic_phonemes(self):
        """Test basic phoneme conversion."""
        phonemizer = Phonemizer()
        text = "hello"
        result = phonemizer.phonemize(text)
        assert result  # Should return phonemes
        assert isinstance(result, str)

    def test_language_codes(self):
        """Test different language codes."""
        text = "hello"
        us_phonemizer = Phonemizer("a")  # US English
        uk_phonemizer = Phonemizer("b")  # UK English
        us_result = us_phonemizer.phonemize(text)
        uk_result = uk_phonemizer.phonemize(text)
        assert us_result != uk_result  # Should differ

    def test_invalid_language(self):
        """Test invalid language code."""
        with pytest.raises(ValueError):
            Phonemizer("invalid")

    def test_punctuation(self):
        """Test punctuation handling."""
        phonemizer = Phonemizer()
        text = "Hello, world!"
        result = phonemizer.phonemize(text)
        assert result  # Should handle punctuation


class TestTokenizer:
    """Test token generation."""

    def test_basic_tokenization(self):
        """Test basic token generation."""
        tokenizer = Tokenizer()
        phonemes = "həˈloʊ"
        tokens = tokenizer.tokenize(phonemes)
        assert tokens
        assert all(isinstance(t, int) for t in tokens)

    def test_vocabulary(self):
        """Test vocabulary coverage."""
        vocab = get_vocab()
        symbols = get_symbols()
        assert len(vocab) > 0
        assert len(symbols) > 0
        assert all(s in vocab for s in symbols)

    def test_padding(self):
        """Test sequence padding."""
        tokenizer = Tokenizer()
        phonemes = "test"
        tokens = tokenizer.tokenize(phonemes)
        padded = tokenizer.pad_sequence(tokens, max_length=10)
        assert len(padded) == 10
        assert padded[-1] == tokenizer.pad_id

    def test_decode(self):
        """Test token decoding."""
        tokenizer = Tokenizer()
        original = "test"
        tokens = tokenizer.tokenize(original)
        decoded = tokenizer.decode(tokens)
        assert decoded.replace(" ", "") == original.replace(" ", "")


class TestChunker:
    """Test text chunking."""

    def test_basic_chunking(self):
        """Test basic text splitting."""
        chunker = TextChunker(max_length=10)
        text = "This is a test sentence."
        chunks = list(chunker.split_text(text))
        assert chunks
        assert all(len(c) <= 10 for c in chunks)

    def test_sentence_boundaries(self):
        """Test sentence boundary handling."""
        chunker = TextChunker()
        text = "First sentence. Second sentence."
        chunks = list(chunker.split_text(text))
        assert len(chunks) == 2
        assert chunks[0].endswith(".")
        assert chunks[1].endswith(".")

    def test_long_words(self):
        """Test long word handling."""
        chunker = TextChunker(max_length=5)
        text = "supercalifragilisticexpialidocious"
        chunks = list(chunker.split_text(text))
        assert chunks  # Should split even without spaces

    def test_empty_input(self):
        """Test empty input handling."""
        chunker = TextChunker()
        with pytest.raises(ValueError):
            list(chunker.split_text(""))
        with pytest.raises(ValueError):
            list(chunker.split_text("   "))


class TestIntegration:
    """Integration tests for text processing."""

    def test_complete_pipeline(self):
        """Test complete text processing pipeline."""
        text = "Hello, world! This is a test."
        token_sequences = process_text(text)
        assert token_sequences
        assert all(isinstance(seq, list) for seq in token_sequences)
        assert all(
            isinstance(token, int)
            for seq in token_sequences
            for token in seq
        )

    def test_error_propagation(self):
        """Test error handling through pipeline."""
        with pytest.raises(ValueError):
            process_text("")

    def test_long_text(self):
        """Test processing of long text."""
        text = "This is a longer text. " * 10
        token_sequences = process_text(text)
        assert len(token_sequences) > 1  # Should be split into chunks

    def test_special_cases(self):
        """Test special case handling."""
        cases = [
            "Test with numbers 123",
            "Test with symbols @#$",
            "Test with... ellipsis",
            'Test with "quotes"'
        ]
        for text in cases:
            token_sequences = process_text(text)
            assert token_sequences  # Should handle all cases
