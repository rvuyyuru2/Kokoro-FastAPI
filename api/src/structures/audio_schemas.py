"""Audio processing configuration schemas."""

from dataclasses import dataclass
from enum import Enum


class PadStrategy(Enum):
    """Padding strategy options."""
    
    SILENCE = "silence"  # Pure silence
    FADE = "fade"  # Fade to silence
    NOISE = "noise"  # Low-level noise
    REPEAT = "repeat"  # Repeat edge content


@dataclass
class FormatConfig:
    """Audio format configuration."""
    
    # MP3 settings
    mp3_bitrate_mode: str = "CONSTANT"
    mp3_compression: float = 0.0
    
    # Opus settings
    opus_compression: float = 0.0
    
    # FLAC settings
    flac_compression: float = 0.0


@dataclass
class NormConfig:
    """Audio normalization configuration."""
    
    target_db: float = -20.0  # Target dB level
    chunk_trim_ms: int = 50  # Milliseconds to trim from chunk edges
    sample_rate: int = 24000  # Audio sample rate
    min_level: float = 1e-8  # Minimum level to prevent log(0)
    window_size: int = 2048  # Analysis window size


@dataclass
class PadConfig:
    """Padding configuration."""
    
    strategy: PadStrategy = PadStrategy.SILENCE
    min_silence_ms: float = 100.0  # Minimum silence duration
    noise_level: float = 0.001  # Noise amplitude for noise strategy
    fade_ms: float = 50.0  # Fade duration for fade strategy
    sample_rate: int = 24000  # Audio sample rate
    silence_threshold: float = 0.01  # Amplitude threshold for silence


@dataclass
class EffectConfig:
    """Effect configuration."""
    
    # Fade settings
    fade_samples: int = 128  # Default fade length
    fade_curve: str = "linear"  # linear, exponential, cosine
    
    # Compression settings
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 5.0
    release_ms: float = 50.0
    
    # EQ settings
    low_shelf_gain: float = 0.0
    high_shelf_gain: float = 0.0


@dataclass
class AudioConfig:
    """Combined audio processing configuration."""
    
    format: FormatConfig = FormatConfig()
    norm: NormConfig = NormConfig()
    pad: PadConfig = PadConfig()
    effect: EffectConfig = EffectConfig()