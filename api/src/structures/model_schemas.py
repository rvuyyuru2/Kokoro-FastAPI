"""Model configuration schemas."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Base model configuration."""
    cache_voices: bool = True
    voice_cache_size: int = 10


@dataclass
class ONNXConfig:
    """ONNX model configuration."""
    optimization_level: str = "all"  # all, basic, none
    num_threads: int = 4
    inter_op_threads: int = 4
    execution_mode: str = "parallel"  # parallel, sequential
    memory_pattern: bool = True
    arena_extend_strategy: str = "kNextPowerOfTwo"


@dataclass
class ONNXGPUConfig(ONNXConfig):
    """ONNX GPU model configuration."""
    num_streams: int = 1  # Number of CUDA streams for parallel inference
    max_streams: int = 4  # Maximum allowed streams
    stream_sync_timeout: float = 10.0  # Seconds to wait for stream sync
    memory_threshold: float = 4.0  # GB
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        super().__post_init__()
        # Ensure num_streams is within bounds
        self.num_streams = max(1, min(self.num_streams, self.max_streams))


@dataclass
class PyTorchConfig:
    """Base PyTorch model configuration."""
    memory_threshold: float = 4.0  # GB
    num_threads: int = 4  # Number of CPU threads for parallel processing


@dataclass
class PyTorchCPUConfig(PyTorchConfig):
    """PyTorch CPU model configuration."""
    inter_op_threads: int = 4  # Number of threads for inter-op parallelism
    intra_op_threads: int = 4  # Number of threads for intra-op parallelism


@dataclass
class PyTorchGPUConfig(PyTorchConfig):
    """PyTorch GPU model configuration."""
    num_streams: int = 1  # Number of CUDA streams for parallel inference
    max_streams: int = 4  # Maximum allowed streams
    stream_sync_timeout: float = 10.0  # Seconds to wait for stream sync
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        # Ensure num_streams is within bounds
        self.num_streams = max(1, min(self.num_streams, self.max_streams))