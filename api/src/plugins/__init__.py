"""Plugin system for TTS pipeline extensibility."""

import pluggy

from .hooks import TTSHookSpec, TTSHookImpl

hookspec = pluggy.HookspecMarker("kokoro_tts")
hookimpl = pluggy.HookimplMarker("kokoro_tts")

__all__ = ["TTSHookSpec", "TTSHookImpl", "hookspec", "hookimpl"]