"""Hook specifications and implementations for TTS pipeline."""

from typing import Optional, Dict, Any

import numpy as np
from pluggy import HookspecMarker, HookimplMarker, PluginManager

hookspec = HookspecMarker("kokoro_tts")
hookimpl = HookimplMarker("kokoro_tts")


class TTSHookSpec:
    """Hook specifications for TTS pipeline."""

    @hookspec
    def pre_process_text(self, text: str) -> str:
        """Pre-process text before normalization."""

    @hookspec
    def post_process_text(self, text: str) -> str:
        """Post-process text after normalization."""

    @hookspec
    def pre_process_phonemes(self, phonemes: str) -> str:
        """Pre-process phonemes before tokenization."""

    @hookspec
    def post_process_phonemes(self, phonemes: str) -> str:
        """Post-process phonemes after tokenization."""

    @hookspec
    def pre_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Pre-process audio before normalization."""

    @hookspec
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Post-process audio after normalization."""


class AudioPluginBase:
    """Base class for audio processing plugins."""

    @hookimpl(optionalhook=True)
    def pre_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Optional pre-processing hook."""
        return audio

    @hookimpl(optionalhook=True)
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Optional post-processing hook."""
        return audio


class TextPluginBase:
    """Base class for text processing plugins."""

    @hookimpl(optionalhook=True)
    def pre_process_text(self, text: str) -> str:
        """Optional pre-processing hook."""
        return text

    @hookimpl(optionalhook=True)
    def post_process_text(self, text: str) -> str:
        """Optional post-processing hook."""
        return text


class PhonemePluginBase:
    """Base class for phoneme processing plugins."""

    @hookimpl(optionalhook=True)
    def pre_process_phonemes(self, phonemes: str) -> str:
        """Optional pre-processing hook."""
        return phonemes

    @hookimpl(optionalhook=True)
    def post_process_phonemes(self, phonemes: str) -> str:
        """Optional post-processing hook."""
        return phonemes


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def initialize_plugin_manager() -> PluginManager:
    """Initialize the global plugin manager singleton.
    
    Returns:
        Configured plugin manager instance
    """
    import json
    import importlib
    from pathlib import Path
    from loguru import logger
    
    global _plugin_manager
    
    if _plugin_manager is not None:
        return _plugin_manager

    # Initialize plugin manager
    pm = PluginManager("kokoro_tts")
    pm.add_hookspecs(TTSHookSpec)
    
    # Get plugins directory path
    plugins_dir = Path(__file__).parent
    config_path = plugins_dir / "plugin_config.json"
    
    # Load enabled plugins from config
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                enabled_plugins = config.get("enabled_plugins", [])
                
            logger.info(f"Loading plugins: {enabled_plugins}")
            
            # Import and register each enabled plugin
            for plugin_name in enabled_plugins:
                try:
                    # Import the plugin module
                    plugin_module = importlib.import_module(
                        f".{plugin_name}",
                        package="api.src.plugins"
                    )
                    
                    # Get the plugin instance
                    plugin_instance = getattr(plugin_module, f"{plugin_name}_plugin")
                    
                    # Register the plugin
                    pm.register(plugin_instance)
                    logger.info(f"Successfully loaded plugin: {plugin_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Failed to load plugin config: {str(e)}")
    else:
        logger.warning(f"No plugin config found at {config_path}")
    
    _plugin_manager = pm
    return pm


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance.
    
    Returns:
        Plugin manager instance
        
    Raises:
        RuntimeError: If plugin manager not initialized
    """
    if _plugin_manager is None:
        raise RuntimeError(
            "Plugin manager not initialized. "
            "Call initialize_plugin_manager() first."
        )
    return _plugin_manager