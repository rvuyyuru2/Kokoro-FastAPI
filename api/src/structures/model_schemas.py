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
class ONNXGPUConfig(ONNXConfig):
    """ONNX GPU-specific configuration."""
    
    device_id: int = 0  # CUDA device ID
    gpu_mem_limit: float = 0.7  # Fraction of GPU memory to use
    cudnn_conv_algo_search: str = "EXHAUSTIVE"  # CuDNN convolution algorithm search
    do_copy_in_default_stream: bool = True  # Copy in default CUDA stream


@dataclass
class PyTorchConfig:
    """PyTorch backend configuration."""
    
    device_id: int = 0  # CUDA device ID
    use_fp16: bool = True  # Whether to use FP16 precision
    use_triton: bool = True  # Whether to use Triton for CUDA kernels
    max_batch_size: int = 32  # Maximum batch size for batched inference
    stream_buffer_size: int = 8  # Size of CUDA stream buffer
    memory_threshold: float = 0.8  # Memory threshold for cleanup
    retry_on_oom: bool = True  # Whether to retry on OOM errors
    sync_cuda: bool = True  # Whether to synchronize CUDA operations


@dataclass
class PyTorchCPUConfig(PyTorchConfig):
    """PyTorch CPU-specific configuration."""
    
    num_threads: int = 8  # Number of threads for parallel operations
    pin_memory: bool = True  # Whether to pin memory for faster CPU-GPU transfer


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # General settings
    model_type: str = "pytorch"  # Model type ('pytorch' or 'onnx')
    device_type: str = "auto"  # Device type ('cpu', 'gpu', or 'auto')
    cache_models: bool = True  # Whether to cache loaded models
    cache_voices: bool = True  # Whether to cache voice tensors
    voice_cache_size: int = 10  # Maximum number of cached voices
    
    # Backend-specific configs
    onnx_cpu: ONNXConfig = ONNXConfig()
    onnx_gpu: ONNXGPUConfig = ONNXGPUConfig()
    pytorch_cpu: PyTorchCPUConfig = PyTorchCPUConfig()
    pytorch_gpu: PyTorchConfig = PyTorchConfig()
    
    def get_backend_config(self, backend_type: str):
        """Get configuration for specific backend.
        
        Args:
            backend_type: Backend type ('pytorch_cpu', 'pytorch_gpu', 'onnx_cpu', 'onnx_gpu')
            
        Returns:
            Backend-specific configuration
            
        Raises:
            ValueError: If backend type is invalid
        """
        if backend_type not in {
            'pytorch_cpu', 'pytorch_gpu', 'onnx_cpu', 'onnx_gpu'
        }:
            raise ValueError(f"Invalid backend type: {backend_type}")
            
        return getattr(self, backend_type)