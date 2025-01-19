"""GPU-based ONNX inference backend."""

import gc
import asyncio
from typing import Dict, Optional, List

import numpy as np
import torch
from loguru import logger
from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers
)

from ..core import paths
from ..core.config import settings
from ..structures.model_schemas import ONNXGPUConfig
from .base import BaseModelBackend, ModelState


class ONNXGPUBackend(BaseModelBackend):
    """ONNX-based GPU inference backend."""

    def __init__(self, config: Optional[ONNXGPUConfig] = None):
        """Initialize GPU backend."""
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if "CUDAExecutionProvider" not in get_available_providers():
            raise RuntimeError("CUDA provider not available for ONNX")
            
        self._device = "cuda"
        self._session: Optional[InferenceSession] = None
        self._config = config or ONNXGPUConfig()
        self._state = ModelState.UNINITIALIZED
        
        # Initialize CUDA streams
        self._streams = [
            torch.cuda.Stream()
            for _ in range(self._config.num_streams)
        ]
        self._current_stream = 0

    async def load_model(self, path: str) -> None:
        """Load ONNX model.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            
            logger.info(f"Loading ONNX model: {model_path}")
            
            # Configure session
            options = self._create_session_options()
            provider_options = self._create_provider_options()
            
            # Create session
            self._session = InferenceSession(
                model_path,
                sess_options=options,
                providers=["CUDAExecutionProvider"],
                provider_options=[provider_options]
            )
            
            self._state = ModelState.LOADED
            logger.info("ONNX model loaded successfully")
            
        except Exception as e:
            self._state = ModelState.FAILED
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    async def generate(
        self,
        tokens: List[int],
        voice: torch.Tensor,
        speed: float = 1.0,
        stream: Optional[torch.cuda.Stream] = None
    ) -> np.ndarray:
        """Generate audio using ONNX model.
        
        Args:
            tokens: Input token IDs
            voice: Voice embedding tensor
            speed: Speed multiplier
            stream: Optional CUDA stream for parallel execution
            
        Returns:
            Generated audio samples
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_ready:
            raise RuntimeError("Model not ready for inference")

        try:
            # Check memory and cleanup if needed
            if self._check_memory():
                self._clear_memory()

            # Use provided stream or get next stream from pool
            stream_to_use = stream or self._streams[self._current_stream]
            self._current_stream = (self._current_stream + 1) % len(self._streams)

            with torch.cuda.stream(stream_to_use):
                # Prepare inputs
                tokens_input = np.array([tokens], dtype=np.int64)
                style_input = voice[len(tokens)].cpu().numpy()  # Move to CPU for ONNX
                speed_input = np.full(1, speed, dtype=np.float32)

                # Run inference in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._session.run,
                    None,
                    {
                        "tokens": tokens_input,
                        "style": style_input,
                        "speed": speed_input
                    }
                )

                # Ensure stream sync
                stream_to_use.synchronize()
                return result[0]
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def _create_session_options(self) -> SessionOptions:
        """Create ONNX session options.
        
        Returns:
            Configured session options
        """
        options = SessionOptions()
        
        # Set optimization level
        if self._config.optimization_level == "all":
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        elif self._config.optimization_level == "basic":
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:
            options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        
        # Configure threading
        options.intra_op_num_threads = self._config.num_threads
        options.inter_op_num_threads = self._config.inter_op_threads
        
        # Set execution mode
        options.execution_mode = (
            ExecutionMode.ORT_PARALLEL
            if self._config.execution_mode == "parallel"
            else ExecutionMode.ORT_SEQUENTIAL
        )
        
        # Configure memory optimization
        options.enable_mem_pattern = self._config.memory_pattern
        
        return options

    def _create_provider_options(self) -> Dict:
        """Create CUDA provider options.
        
        Returns:
            Provider configuration
        """
        return {
            "CUDAExecutionProvider": {
                "arena_extend_strategy": self._config.arena_extend_strategy,
                "gpu_mem_limit": int(self._config.memory_threshold * 1024 * 1024 * 1024),  # Convert GB to bytes
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True
            }
        }

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1e9
            return memory_gb > self._config.memory_threshold
        return False

    def _clear_memory(self) -> None:
        """Clear GPU memory."""
        if torch.cuda.is_available():
            # Wait for all streams to complete
            for stream in self._streams:
                stream.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        
    def _cleanup_resources(self) -> None:
        """Clean up ONNX resources."""
        if self._session is not None:
            # Wait for all streams to complete
            for stream in self._streams:
                stream.synchronize()
            del self._session
            self._session = None
        super()._cleanup_resources()

    async def warmup(self) -> None:
        """Run model warmup.
        
        Raises:
            RuntimeError: If warmup fails
        """
        if not self.is_loaded:
            raise RuntimeError("Cannot warmup - model not loaded")
            
        # Model warmup is handled by model manager
        self._state = ModelState.WARMED_UP
        logger.info("ONNX model warmup completed")

    @property
    def state(self) -> ModelState:
        """Get current model state."""
        if self._session is None:
            return ModelState.UNINITIALIZED
        return self._state