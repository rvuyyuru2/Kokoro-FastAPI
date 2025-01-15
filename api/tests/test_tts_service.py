"""Tests for TTSService"""

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch, AsyncMock

import numpy as np
import pytest
import torch
from onnxruntime import InferenceSession

from api.src.core.config import settings
from api.src.services.tts_cpu import TTSCPUModel
from api.src.services.tts_gpu import TTSGPUModel
from api.src.services.tts_model import TTSModel
from api.src.services.tts_service import TTSService


@pytest.fixture
def tts_service(monkeypatch):
    """Create a TTSService instance for testing"""
    # Mock TTSModel initialization
    mock_model = MagicMock()
    mock_model.generate_from_tokens = MagicMock(return_value=np.zeros(48000))
    mock_model.process_text = MagicMock(return_value=("mock phonemes", [1, 2, 3]))

    # Set up model instance
    monkeypatch.setattr("api.src.services.tts_model.TTSModel._instance", mock_model)
    monkeypatch.setattr(
        "api.src.services.tts_model.TTSModel.get_instance",
        MagicMock(return_value=mock_model),
    )
    monkeypatch.setattr(
        "api.src.services.tts_model.TTSModel.get_device", MagicMock(return_value="cpu")
    )

    return TTSService()


@pytest.fixture
def sample_audio():
    """Generate a simple sine wave for testing"""
    sample_rate = 24000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


def test_audio_to_bytes(tts_service, sample_audio):
    """Test converting audio tensor to bytes"""
    audio_bytes = tts_service._audio_to_bytes(sample_audio)
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0


@pytest.mark.asyncio
async def test_generate_audio_empty_text(tts_service):
    """Test generating audio with empty text"""
    with pytest.raises(ValueError):
        await tts_service._generate_audio("", "af", 1.0)


def test_save_audio(tts_service, sample_audio, tmp_path):
    """Test saving audio to file"""
    output_path = os.path.join(tmp_path, "test_output.wav")
    tts_service._save_audio(sample_audio, output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


@pytest.mark.asyncio
async def test_combine_voices_validation(tts_service):
    """Test voice combination input validation"""
    with pytest.raises(ValueError):
        await tts_service.combine_voices([])  # Empty list
    
    with pytest.raises(ValueError):
        await tts_service.combine_voices(["single_voice"])  # Single voice
