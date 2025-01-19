"""Base classes for TTS pipeline."""

import os
from abc import abstractmethod
from typing import Any, Optional, AsyncIterator, List, Union

import numpy as np
from loguru import logger

from ..structures.pipeline_protocols import Pipeline
from ..audio_processing import (
    AudioProcessor,
    get_processor as get_audio_processor
)
from ..inference import ModelManager, get_manager
from ..text_processing import process_text


class BasePipeline(Pipeline):
    """Base TTS pipeline implementation."""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        audio_processor: Optional[AudioProcessor] = None,
        voices_dir: str = "voices",
        plugin_manager: Optional[Any] = None
    ):
        """Initialize pipeline.
        
        Args:
            model_manager: Optional model manager
            audio_processor: Optional audio processor
            voices_dir: Voice directory path
            plugin_manager: Optional plugin manager
        """
        self._model_manager = model_manager or get_manager()
        self._audio_processor = audio_processor or get_audio_processor()
        self._plugin_manager = plugin_manager
        self._voices_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            voices_dir
        )

    def _apply_text_preprocessing(self, text: str) -> str:
        """Apply text pre-processing plugins.
        
        Args:
            text: Raw input text
            
        Returns:
            Pre-processed text
        """
        if not self._plugin_manager:
            return text
            
        processed = text
        for result in self._plugin_manager.hook.pre_process_text(text=processed):
            if result is not None:
                processed = result
        return processed

    def _apply_audio_postprocessing(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio post-processing plugins.
        
        Args:
            audio: Raw audio samples
            
        Returns:
            Processed audio samples
        """
        if not self._plugin_manager:
            return audio
            
        processed = audio
        for result in self._plugin_manager.hook.post_process_audio(audio=processed):
            if result is not None:
                processed = result
        return processed

    async def get_voice_path(self, voice: str) -> Optional[str]:
        """Get path to voice file.
        
        Args:
            voice: Voice ID
            
        Returns:
            Voice file path or None if not found
        """
        path = os.path.join(self._voices_dir, f"{voice}.pt")
        return path if os.path.exists(path) else None

    async def cleanup_voice(self, voice_path: str) -> None:
        """Cleanup voice resources.
        
        Args:
            voice_path: Path to voice file
        """
        # No cleanup needed for base implementation
        pass

    @abstractmethod
    async def process(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav",
        stream: bool = False
    ) -> Union[AsyncIterator[bytes], bytes]:
        """Process text to speech.
        
        Args:
            text: Input text
            voice: Voice ID
            speed: Speed multiplier
            format: Output format
            stream: Whether to stream output
            
        Returns:
            Audio chunks for streaming pipeline, complete audio for whole file pipeline
        """
        pass