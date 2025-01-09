import torch
from ..core.config import settings

from loguru import logger

def cleanup(verbose: bool = False):
    """Clean up system resources like CUDA cache if enabled"""
    if torch.cuda.is_available():
        if verbose:
            allocated_0, reserved_0 = log_gpu_memory(verbose=False)
        if settings.clear_cuda_cache:
            torch.cuda.empty_cache()
        if verbose:
            allocated_1, reserved_1 = log_gpu_memory(verbose=False)
            logger.debug(f"""
                Freed GPU Memory: {allocated_0 - allocated_1:.1f}MB allocated, {reserved_0 - reserved_1:.1f}MB reserved"""
            )

def log_gpu_memory(msg: str = "", verbose: bool = True):
    """Log GPU memory usage and return allocated and reserved"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        if verbose:
            logger.debug(f"[{msg}]: GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        return allocated, reserved
