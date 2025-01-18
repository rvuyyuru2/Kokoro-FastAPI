"""Hook specifications and implementations for TTS pipeline."""

from typing import Optional

import numpy as np
from pluggy import HookspecMarker, HookimplMarker

from ..structures.pipeline_protocols import Pipeline

hookspec = HookspecMarker("kokoro_tts")
hookimpl = HookimplMarker("kokoro_tts")


class TTSHookSpec:
    """Hook specifications for TTS pipeline."""

    @hookspec
    def pre_process_text(self, text: str) -> str:
        """Pre-process text before normalization.
        
        Args:
            text: Raw input text
            
        Returns:
            Pre-processed text
        """

    @hookspec
    def post_process_text(self, text: str) -> str:
        """Post-process text after normalization.
        
        Args:
            text: Normalized text
            
        Returns:
            Post-processed text
        """

    @hookspec
    def pre_process_phonemes(self, phonemes: str) -> str:
        """Pre-process phonemes before tokenization.
        
        Args:
            phonemes: Raw phoneme string
            
        Returns:
            Pre-processed phonemes
        """

    @hookspec
    def post_process_phonemes(self, phonemes: str) -> str:
        """Post-process phonemes after tokenization.
        
        Args:
            phonemes: Tokenized phonemes
            
        Returns:
            Post-processed phonemes
        """

    @hookspec
    def pre_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Pre-process audio before normalization.
        
        Args:
            audio: Raw audio samples
            
        Returns:
            Pre-processed audio
        """

    @hookspec
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Post-process audio after normalization.
        
        Args:
            audio: Normalized audio samples
            
        Returns:
            Post-processed audio
        """

    @hookspec
    def customize_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """Customize pipeline configuration.
        
        Args:
            pipeline: Base pipeline instance
            
        Returns:
            Customized pipeline
        """


class TTSHookImpl:
    """Default implementations of TTS hooks."""

    @hookimpl
    def pre_process_text(self, text: str) -> str:
        """Default text pre-processing."""
        return text

    @hookimpl
    def post_process_text(self, text: str) -> str:
        """Default text post-processing."""
        return text

    @hookimpl
    def pre_process_phonemes(self, phonemes: str) -> str:
        """Default phoneme pre-processing."""
        return phonemes

    @hookimpl
    def post_process_phonemes(self, phonemes: str) -> str:
        """Default phoneme post-processing."""
        return phonemes

    @hookimpl
    def pre_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Default audio pre-processing."""
        return audio

    @hookimpl
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Default audio post-processing."""
        return audio

    @hookimpl
    def customize_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """Default pipeline customization."""
        return pipeline


def create_plugin_manager():
    """Create and configure plugin manager.
    
    Returns:
        Configured plugin manager
    """
    import pluggy
    
    pm = pluggy.PluginManager("kokoro_tts")
    pm.add_hookspecs(TTSHookSpec)
    pm.register(TTSHookImpl())
    
    return pm