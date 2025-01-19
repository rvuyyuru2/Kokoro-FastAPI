"""Helium voice effect plugin that increases pitch."""
import numpy as np
from scipy import signal
from loguru import logger

from .hooks import hookimpl, AudioPluginBase


class HeliumPlugin(AudioPluginBase):
    """Plugin to increase the pitch of the audio in post-processing."""

    @hookimpl
    def post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Increase the pitch of the audio by resampling.
        
        Args:
            audio: Input audio samples as numpy array
            
        Returns:
            Pitch-shifted audio samples
        """
        logger.info("Helium plugin: Processing audio...")
        processed = audio.astype(np.float32)
        num_samples = len(processed)
        
        # high pitch helium effect
        pitch_factor = 2.0
        new_num_samples = int(num_samples / pitch_factor)
        
        logger.info(f"Resampling from {num_samples} to {new_num_samples} samples")
        pitched_audio = signal.resample(processed, new_num_samples)

        warble_factor = 0.5
        warble = np.sin(np.linspace(0, warble_factor * np.pi, new_num_samples)).astype(np.float32)
        pitched_audio = pitched_audio * warble
        logger.info(f"Resampled audio shape: {pitched_audio.shape}")

        if len(pitched_audio) > num_samples:
            pitched_audio = pitched_audio[:num_samples]
        elif len(pitched_audio) < num_samples:
            pitched_audio = np.pad(pitched_audio, (0, num_samples - len(pitched_audio)))
            
        return pitched_audio 


# Create plugin instance
helium_plugin = HeliumPlugin()