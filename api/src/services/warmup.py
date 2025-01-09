import os
from typing import List, Tuple
import torch
from loguru import logger

from .tts_service import TTSService
from .tts_model import TTSModel
from ..utils.housekeeping import log_gpu_memory
from ..core.config import settings

class WarmupService:
    """Service for warming up TTS models and voice caches"""
    
    def __init__(self, tts_service: TTSService = None):
        log_gpu_memory("Baseline Startup VRAM Usage")
        self.tts_service = tts_service or TTSService()
        
    def load_voices(self) -> List[Tuple[str, torch.Tensor]]:
        """Load and cache voices up to LRU limit"""
        voice_files = sorted(
            [f for f in os.listdir(TTSModel.VOICES_DIR) if f.endswith(".pt")],
            key=len
        )
        
        n_cache_voices = settings.n_cache_voices
        loaded_voices = []
        for voice_file in voice_files[:n_cache_voices]:
            try:
                voice_path = os.path.join(TTSModel.VOICES_DIR, voice_file)
                voicepack = self.tts_service._load_voice(voice_path)
                loaded_voices.append((voice_file[:-3], voicepack))
            except Exception as e:
                logger.error(f"Failed to load voice {voice_file}: {e}")
        logger.info(f"Pre-loaded {len(loaded_voices)} voices into cache")

        return loaded_voices
        
    async def warmup_voices(self, warmup_text: str, loaded_voices: List[Tuple[str, torch.Tensor]]):
        """Warm up voice inference and streaming"""
        n_warmups = settings.n_warmups
        for voice_name, _ in loaded_voices[:n_warmups]:
            try:
                logger.info(f"Running warmup inference on voice {voice_name}")
                async for _ in self.tts_service.generate_audio_stream(
                    warmup_text,
                    voice_name,
                    1.0,
                    "pcm"
                ):
                    pass  # Process all chunks to properly warm up
                logger.info(f"Completed warmup for voice {voice_name}")
            except Exception as e:
                logger.warning(f"Warmup failed for voice {voice_name}: {e}")
        log_gpu_memory("After warmup")
