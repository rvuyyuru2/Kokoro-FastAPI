"""Model configuration schemas."""

from dataclasses import dataclass


@dataclass
class ONNXConfig:
    """ONNX runtime configuration."""
    
    num_threads: int = 8  # Number of threads for parallel operations
    inter_op_threads: int = 4  # Number of threads for operator parallelism
    execution_mode: str = "parallel"  # ONNX execution mode
    optimization_level: str = "all"  # ONNX optimization level
    memory_pattern: bool = True  # Enable memory pattern optimization
    arena_extend_strategy: str = "kNextPowerOfTwo"  # Memory arena strategy


@dataclass
class GPUConfig:
    """GPU backend configuration."""
    
    device_id: int = 0  # CUDA device ID
    use_fp16: bool = True  # Whether to use FP16 precision
    use_triton: bool = True  # Whether to use Triton for CUDA kernels
    max_batch_size: int = 32  # Maximum batch size for batched inference
    stream_buffer_size: int = 8  # Size of CUDA stream buffer


@dataclass
class ModelConfig:
    """Model configuration."""
    
    prefer_gpu: bool = True  # Whether to prefer GPU when available
    cache_models: bool = True  # Whether to cache loaded models
    cache_voices: bool = True  # Whether to cache voice tensors
    voice_cache_size: int = 10  # Maximum number of cached voices
    
    # Backend-specific configs
    onnx: ONNXConfig = ONNXConfig()
    gpu: GPUConfig = GPUConfig()