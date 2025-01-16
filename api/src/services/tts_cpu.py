import os

import numpy as np
import torch
from loguru import logger
from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)

from ..core.config import settings
from .text_processing import phonemize, tokenize
from .tts_base import TTSBaseModel


class TTSCPUModel(TTSBaseModel):
    def __init__(self):
        super().__init__()
        self._onnx_session = None
        self._device = "cpu"

    def get_instance(self):
        """Get the model instance"""
        if self._onnx_session is None:
            raise RuntimeError("ONNX model not initialized. Call initialize() first.")
        return self._onnx_session

    def initialize(self, model_dir: str, model_path: str = None):
        """Initialize ONNX model for CPU inference"""
        if self._onnx_session is None:
            try:
                # Try loading ONNX model
                if not os.path.exists(model_path):
                    logger.error(f"ONNX model not found at {model_path}")
                    raise RuntimeError(f"ONNX model not found at {model_path}")

                logger.info(f"Loading ONNX model from {model_path}")

                # Configure ONNX session for optimal performance
                session_options = SessionOptions()

                # Set optimization level
                if settings.onnx_optimization_level == "all":
                    session_options.graph_optimization_level = (
                        GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                elif settings.onnx_optimization_level == "basic":
                    session_options.graph_optimization_level = (
                        GraphOptimizationLevel.ORT_ENABLE_BASIC
                    )
                else:
                    session_options.graph_optimization_level = (
                        GraphOptimizationLevel.ORT_DISABLE_ALL
                    )

                # Configure threading
                session_options.intra_op_num_threads = settings.onnx_num_threads
                session_options.inter_op_num_threads = settings.onnx_inter_op_threads

                # Set execution mode
                session_options.execution_mode = (
                    ExecutionMode.ORT_PARALLEL
                    if settings.onnx_execution_mode == "parallel"
                    else ExecutionMode.ORT_SEQUENTIAL
                )

                # Enable/disable memory pattern optimization
                session_options.enable_mem_pattern = settings.onnx_memory_pattern

                # Configure CPU provider options
                provider_options = {
                    "CPUExecutionProvider": {
                        "arena_extend_strategy": settings.onnx_arena_extend_strategy,
                        "cpu_memory_arena_cfg": "cpu:0",
                    }
                }

                session = InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                    provider_options=[provider_options],
                )
                self._onnx_session = session
                return session
            except Exception as e:
                logger.error(f"Failed to initialize ONNX model: {e}")
                return None
        return self._onnx_session

    def process_text(self, text: str, language: str) -> tuple[str, list[int]]:
        """Process text into phonemes and tokens

        Args:
            text: Input text
            language: Language code

        Returns:
            tuple[str, list[int]]: Phonemes and token IDs
        """
        phonemes = phonemize(text, language)
        tokens = tokenize(phonemes)
        tokens = [0] + tokens + [0]  # Add start/end tokens
        return phonemes, tokens

    def generate_from_text(
        self, text: str, voicepack: torch.Tensor, language: str, speed: float
    ) -> tuple[np.ndarray, str]:
        """Generate audio from text

        Args:
            text: Input text
            voicepack: Voice tensor
            language: Language code
            speed: Speed factor

        Returns:
            tuple[np.ndarray, str]: Generated audio samples and phonemes
        """
        if self._onnx_session is None:
            raise RuntimeError("ONNX model not initialized")

        # Process text
        phonemes, tokens = self.process_text(text, language)

        # Generate audio
        audio = self.generate_from_tokens(tokens, voicepack, speed)

        return audio, phonemes

    def generate_from_tokens(
        self, tokens: list[int], voicepack: torch.Tensor, speed: float
    ) -> np.ndarray:
        """Generate audio from tokens

        Args:
            tokens: Token IDs
            voicepack: Voice tensor
            speed: Speed factor

        Returns:
            np.ndarray: Generated audio samples
        """
        if self._onnx_session is None:
            raise RuntimeError("ONNX model not initialized")

        # Pre-allocate and prepare inputs
        tokens_input = np.array([tokens], dtype=np.int64)
        style_input = voicepack[
            len(tokens) - 2
        ].numpy()  # Already has correct dimensions
        speed_input = np.full(
            1, speed, dtype=np.float32
        )  # More efficient than ones * speed

        # Run inference with optimized inputs
        result = self._onnx_session.run(
            None, {"tokens": tokens_input, "style": style_input, "speed": speed_input}
        )
        return result[0]
