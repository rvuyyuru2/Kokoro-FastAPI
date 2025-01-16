from abc import ABC
from .service_interfaces import TTSServiceProtocol


class TTSServiceInterface(ABC, TTSServiceProtocol):
    """Interface defining TTS service capabilities needed by strategies"""
    pass