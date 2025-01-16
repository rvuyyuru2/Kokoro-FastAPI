import re
from typing import Generator, Dict, Any, List, Tuple
import numpy as np

from ....core.config import settings
from ..strategy_interface import ChunkingStrategy


class DynamicChunkStrategy(ChunkingStrategy):
    """Strategy for splitting text using dynamic analysis of content"""
    
    def __init__(self):
        self._config = {
            'max_chunk_size': settings.max_chunk_size,
            'min_chunk_size': 50,  # Minimum chunk size for better consistency
            'ideal_chunk_size': 150,  # Target chunk size
            'boundary_weights': {
                'sentence': 1.0,    # Weight for sentence boundaries
                'clause': 0.8,      # Weight for clause boundaries (;:)
                'phrase': 0.6,      # Weight for phrase boundaries (,)
                'word': 0.2,        # Weight for word boundaries
            }
        }
    
    @property
    def name(self) -> str:
        return "dynamic"
        
    def _score_boundary(self, text: str, pos: int) -> float:
        """Score a potential boundary position
        
        Args:
            text: Full text being analyzed
            pos: Position to score
            
        Returns:
            Float score (0-1) indicating boundary quality
        """
        if pos <= 0 or pos >= len(text):
            return 0.0
            
        char_before = text[pos - 1]
        weights = self._config['boundary_weights']
        
        # Check boundary types
        if char_before in '.!?':
            return weights['sentence']
        elif char_before in ';:':
            return weights['clause']
        elif char_before == ',':
            return weights['phrase']
        elif char_before == ' ':
            return weights['word']
            
        return 0.0
        
    def _find_best_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best boundary position in a range
        
        Args:
            text: Text to analyze
            start: Start position to search from
            end: End position to search to
            
        Returns:
            Position of best boundary
        """
        best_pos = start
        best_score = -1
        
        for pos in range(start, end):
            score = self._score_boundary(text, pos)
            
            # Adjust score based on position relative to ideal chunk size
            ideal_size = self._config['ideal_chunk_size']
            size_diff = abs(pos - start - ideal_size)
            position_penalty = 1.0 / (1.0 + size_diff / ideal_size)
            score *= position_penalty
            
            if score > best_score:
                best_score = score
                best_pos = pos
                
        return best_pos
        
    def _analyze_chunk_candidates(self, text: str) -> List[Tuple[int, float]]:
        """Analyze text for potential chunk boundaries
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (position, score) tuples
        """
        candidates = []
        for i in range(len(text)):
            score = self._score_boundary(text, i)
            if score > 0:
                candidates.append((i, score))
        return candidates
        
    def split_text(self, text: str) -> Generator[str, None, None]:
        """Split text into chunks using dynamic analysis
        
        Args:
            text: Text to split into chunks
            
        Yields:
            Dynamically determined text chunks
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        text = text.strip()
        if not text:
            return
            
        start = 0
        while start < len(text):
            max_chunk = self._config['max_chunk_size']
            min_chunk = self._config['min_chunk_size']
            
            # Determine search range
            search_end = min(start + max_chunk, len(text))
            if search_end - start < min_chunk and search_end < len(text):
                search_end = start + min_chunk
                
            # Find best boundary in range
            if search_end < len(text):
                end = self._find_best_boundary(text, start + min_chunk, search_end)
                if end == start:  # No good boundary found
                    end = search_end
            else:
                end = search_end
                
            chunk = text[start:end].strip()
            if chunk:
                yield chunk
                
            start = end
            
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
        if 'max_chunk_size' in config:
            max_chunk = config['max_chunk_size']
            if not isinstance(max_chunk, int) or max_chunk <= 0:
                raise ValueError("max_chunk_size must be a positive integer")
                
        if 'min_chunk_size' in config:
            min_chunk = config['min_chunk_size']
            if not isinstance(min_chunk, int) or min_chunk <= 0:
                raise ValueError("min_chunk_size must be a positive integer")
                
        if 'ideal_chunk_size' in config:
            ideal_chunk = config['ideal_chunk_size']
            if not isinstance(ideal_chunk, int) or ideal_chunk <= 0:
                raise ValueError("ideal_chunk_size must be a positive integer")
                
        if 'boundary_weights' in config:
            weights = config['boundary_weights']
            if not isinstance(weights, dict):
                raise ValueError("boundary_weights must be a dictionary")
            for key in ['sentence', 'clause', 'phrase', 'word']:
                if key not in weights:
                    raise ValueError(f"boundary_weights missing required key: {key}")
                if not isinstance(weights[key], (int, float)) or weights[key] < 0:
                    raise ValueError(f"boundary_weights[{key}] must be a non-negative number")
                    
        return True