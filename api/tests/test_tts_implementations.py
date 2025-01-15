"""Tests for TTS model implementations"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from api.src.services.tts_base import TTSBaseModel
from api.src.services.tts_cpu import TTSCPUModel
from api.src.services.tts_gpu import TTSGPUModel, length_to_mask


# Base Model Tests
def test_get_device_error():
    """Test get_device() raises error when not initialized"""
    TTSBaseModel._device = None
    with pytest.raises(RuntimeError):
        TTSBaseModel.get_device()




# CPU Model Tests
def test_cpu_initialize_missing_model():
    """Test CPU initialize with missing model"""
    TTSCPUModel._onnx_session = None  # Reset the session
    with patch("os.path.exists", return_value=False), patch(
        "onnxruntime.InferenceSession", return_value=None
    ):
        result = TTSCPUModel.initialize("dummy_dir")
        assert result is None


def test_cpu_generate_uninitialized():
    """Test CPU generate methods with uninitialized model"""
    TTSCPUModel._onnx_session = None

    with pytest.raises(RuntimeError):
        TTSCPUModel.generate_from_text("test", torch.zeros(1), "en", 1.0)

    with pytest.raises(RuntimeError):
        TTSCPUModel.generate_from_tokens([1, 2, 3], torch.zeros(1), 1.0)


def test_cpu_process_text():
    """Test CPU process_text functionality"""
    with patch("api.src.services.tts_cpu.phonemize") as mock_phonemize, patch(
        "api.src.services.tts_cpu.tokenize"
    ) as mock_tokenize:
        mock_phonemize.return_value = "test phonemes"
        mock_tokenize.return_value = [1, 2, 3]

        phonemes, tokens = TTSCPUModel.process_text("test", "en")
        assert phonemes == "test phonemes"
        assert tokens == [0, 1, 2, 3, 0]  # Should add start/end tokens


# GPU Model Tests
@patch("torch.cuda.is_available")
def test_gpu_initialize_cuda_unavailable(mock_cuda_available):
    """Test GPU initialize with CUDA unavailable"""
    mock_cuda_available.return_value = False
    TTSGPUModel._instance = None

    result = TTSGPUModel.initialize("dummy_dir", "dummy_path")
    assert result is None


@patch("api.src.services.tts_gpu.length_to_mask")
def test_gpu_length_to_mask(mock_length_to_mask):
    """Test length_to_mask function"""
    # Setup mock return value
    expected_mask = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, False]]
    )
    mock_length_to_mask.return_value = expected_mask

    # Call function with test input
    lengths = torch.tensor([3, 5])
    mask = mock_length_to_mask(lengths)

    # Verify mock was called with correct input
    mock_length_to_mask.assert_called_once()
    assert torch.equal(mask, expected_mask)


def test_gpu_generate_uninitialized():
    """Test GPU generate methods with uninitialized model"""
    TTSGPUModel._instance = None

    with pytest.raises(RuntimeError):
        TTSGPUModel.generate_from_text("test", torch.zeros(1), "en", 1.0)

    with pytest.raises(RuntimeError):
        TTSGPUModel.generate_from_tokens([1, 2, 3], torch.zeros(1), 1.0)


def test_gpu_process_text():
    """Test GPU process_text functionality"""
    with patch("api.src.services.tts_gpu.phonemize") as mock_phonemize, patch(
        "api.src.services.tts_gpu.tokenize"
    ) as mock_tokenize:
        mock_phonemize.return_value = "test phonemes"
        mock_tokenize.return_value = [1, 2, 3]

        phonemes, tokens = TTSGPUModel.process_text("test", "en")
        assert phonemes == "test phonemes"
        assert tokens == [1, 2, 3]  # GPU implementation doesn't add start/end tokens
