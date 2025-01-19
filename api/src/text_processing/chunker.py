"""Text chunking for TTS preprocessing."""

import re
from typing import Iterator

from ..plugins import hookimpl
from ..core.config import settings

@hookimpl
def pre_process_text(text: str) -> str:
    """Plugin hook for text pre-processing."""
    return text


@hookimpl
def post_process_text(text: str) -> str:
    """Plugin hook for text post-processing."""
    print(f"post_process_text: {text}")
    return text


def split_text(text: str, max_chunk: int = None) -> Iterator[str]:
    """Split text into chunks on natural pause points.

    Args:
        text: Text to split into chunks
        max_chunk: Maximum chunk size

    Yields:
        Text chunks
    """
    # Apply pre-processing hook
    text = pre_process_text(text)

    # Use configured chunk size
    if max_chunk is None:
        max_chunk = settings.max_chunk_size

    # Handle non-string input
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()
    if not text:
        return

    # First try splitting on sentence boundaries
    chunks = re.split(r"(?<=[.!?])\s+", text)
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # If chunk is too long, split on major punctuation
        if len(chunk) > max_chunk:
            words = chunk.split()
            current = []
            length = 0
            print(f"words: {words}")
            for word in words:

                if length + len(word) + 1 > max_chunk and current:
                    # Apply post-processing hook
                    result = post_process_text(" ".join(current))
                    yield result
                    current = []
                    length = 0
                current.append(word)
                length += len(word) + 1
                
            if current:
                # Apply post-processing hook
                result = post_process_text(" ".join(current))
                yield result
        else:
            # Apply post-processing hook
            chunk = post_process_text(chunk)
            yield chunk