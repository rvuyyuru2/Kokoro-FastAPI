"""Plugin system for TTS pipeline extensibility."""

from .hooks import (
    TTSHookSpec,
    AudioPluginBase,
    TextPluginBase,
    PhonemePluginBase,
    hookspec,
    hookimpl,
    initialize_plugin_manager,
    get_plugin_manager
)

__all__ = [
    "TTSHookSpec",
    "AudioPluginBase",
    "TextPluginBase", 
    "PhonemePluginBase",
    "hookspec",
    "hookimpl",
    "initialize_plugin_manager",
    "get_plugin_manager"
]