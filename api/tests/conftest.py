"""Test configuration and fixtures."""

import os
from typing import AsyncIterator, Iterator

import numpy as np
import pytest
import torch

from ..audio_processing import (
    AudioConfig,
    AudioProcessor,
    FormatConfig,
    NormConfig,
    PadConfig,
    EffectConfig
)
from ..inference import (
    ModelConfig,
    ModelManager,
    ONNXConfig,
    GPUConfig
)
from ..services import ServiceConfig, TTSService
from ..text_processing import process_text


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return "Hello, world! This is a test."


@pytest.fixture
def sample_tokens() -> list[int]:
    """Sample token sequence for testing."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Sample audio data for testing."""
    return np.random.randn(24000)  # 1 second at 24kHz


@pytest.fixture
def sample_voice() -> torch.Tensor:
    """Sample voice embedding for testing."""
    return torch.randn(256)  # Example embedding size


@pytest.fixture
def audio_config() -> AudioConfig:
    """Audio processing configuration."""
    return AudioConfig(
        format=FormatConfig(
            mp3_compression=0.0,
            opus_compression=0.0
        ),
        norm=NormConfig(
            target_db=-20.0,
            chunk_trim_ms=50
        ),
        pad=PadConfig(
            min_silence_ms=100.0,
            noise_level=0.001
        ),
        effect=EffectConfig(
            fade_samples=128,
            threshold_db=-20.0
        )
    )


@pytest.fixture
def model_config() -> ModelConfig:
    """Model configuration."""
    return ModelConfig(
        prefer_gpu=False,  # Use CPU for testing
        cache_models=True,
        cache_voices=True,
        voice_cache_size=10,
        onnx=ONNXConfig(
            optimization_level="basic",
            num_threads=1
        ),
        gpu=GPUConfig(
            memory_threshold=2.0,
            retry_on_oom=True
        )
    )


@pytest.fixture
def service_config(
    audio_config: AudioConfig,
    model_config: ModelConfig
) -> ServiceConfig:
    """Service configuration."""
    return ServiceConfig(
        model=model_config,
        audio=audio_config,
        voices_dir="test_voices",
        output_dir="test_output"
    )


@pytest.fixture
def audio_processor(audio_config: AudioConfig) -> AudioProcessor:
    """Audio processor instance."""
    return AudioProcessor(audio_config)


@pytest.fixture
def model_manager(model_config: ModelConfig) -> ModelManager:
    """Model manager instance."""
    return ModelManager(model_config)


@pytest.fixture
def tts_service(service_config: ServiceConfig) -> TTSService:
    """TTS service instance."""
    return TTSService(service_config)


@pytest.fixture
def text_processor() -> Iterator[str]:
    """Process sample text into chunks."""
    text = "This is a test sentence. And another one."
    yield from process_text(text)


@pytest.fixture
async def streaming_chunks(
    tts_service: TTSService,
    sample_text: str
) -> AsyncIterator[bytes]:
    """Generate streaming audio chunks."""
    async for chunk in tts_service.generate_stream(
        sample_text,
        "test_voice",
        speed=1.0,
        output_format="wav"
    ):
        yield chunk


@pytest.fixture(scope="session")
def test_dir(tmp_path_factory) -> str:
    """Create and clean up test directory."""
    test_dir = tmp_path_factory.mktemp("tts_test")
    os.makedirs(os.path.join(test_dir, "voices"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "output"), exist_ok=True)
    return str(test_dir)


@pytest.fixture(autouse=True)
def cleanup_test_files(test_dir: str) -> None:
    """Clean up test files after each test."""
    yield
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith((".wav", ".mp3", ".opus", ".flac", ".pt")):
                os.remove(os.path.join(root, file))
