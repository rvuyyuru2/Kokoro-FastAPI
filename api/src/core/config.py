from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8880

    # TTS Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af"
    model_dir: str = "models"  # Base directory for model files, relative to project root

    pytorch_model_path: str = "kokoro-v0_19.pth"
    onnx_model_path: str = "kokoro-v0_19.onnx"

    voices_dir: str = "voices"  # Relative to project root
    sample_rate: int = 24000
    max_chunk_size: int = 300  # Maximum size of text chunks for processing
    gap_trim_ms: int = 250  # Amount to trim from streaming chunk ends in milliseconds

    # ONNX Optimization Settings
    onnx_num_threads: int = 4  # Number of threads for intra-op parallelism
    onnx_inter_op_threads: int = 4  # Number of threads for inter-op parallelism
    onnx_execution_mode: str = "parallel"  # parallel or sequential
    onnx_optimization_level: str = "all"  # all, basic, or disabled
    onnx_memory_pattern: bool = True  # Enable memory pattern optimization
    onnx_arena_extend_strategy: str = "kNextPowerOfTwo"  # Memory allocation strategy

    # Model Management Settings
    max_model_instances: int = 2  # Maximum number of concurrent model instances
    model_request_timeout: float = 30.0  # Timeout for model requests in seconds
    model_health_check_interval: float = 60.0  # Interval between health checks
    model_max_requests: int = 1000  # Recycle instance after this many requests
    model_queue_size: int = 100  # Maximum size of pending request queue
    cuda_stream_per_instance: bool = True  # Use separate CUDA streams per instance

    class Config:
        env_file = ".env"


settings = Settings()
