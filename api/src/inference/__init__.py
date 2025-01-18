"""Inference module for model management and generation."""

from .base import ModelBackend
from .model_manager import ModelManager, get_manager
from .cpu import CPUBackend
from .gpu import GPUBackend
from ..structures.model_schemas import ModelConfig, ONNXConfig, GPUConfig

__all__ = [
    "ModelBackend",
    "ModelManager",
    "CPUBackend",
    "GPUBackend",
    "ModelConfig",
    "ONNXConfig",
    "GPUConfig",
    "get_manager"
]