import os
import asyncio
from typing import List, Type, Optional
import torch

from loguru import logger
from ..core.config import settings
from .tts_base import TTSBaseModel


class ModelManager:
    """Manages a pool of model instances for concurrent processing"""
    
    def __init__(
        self, 
        model_class: Type[TTSBaseModel]
    ):
        """
        Initialize ModelManager for managing multiple model instances
        
        Args:
            model_class: The TTSBaseModel subclass to instantiate
        """
        self._model_class = model_class
        self._max_instances = settings.max_model_instances
        
        # Instance pool with locking
        self._instances: List[TTSBaseModel] = []
        self._instance_lock = asyncio.Lock()
        
        # CUDA streams for GPU models
        self._cuda_streams = {}
        if settings.cuda_stream_per_instance and torch.cuda.is_available():
            try:
                for i in range(self._max_instances):
                    self._cuda_streams[i] = torch.cuda.Stream()
                logger.info(f"Created {self._max_instances} CUDA streams for parallel execution")
            except Exception as e:
                logger.error(f"Failed to create CUDA streams: {e}")
                self._cuda_streams = {}  # Fall back to default stream

    async def get_model(self, model_name: str = None) -> TTSBaseModel:
        """
        Retrieve an available model instance, creating one if needed
        
        Args:
            model_name: Optional name of the model to load. If None, uses default model.
            
        Returns:
            An available TTSBaseModel instance
        """
        async with self._instance_lock:
            # Find an available instance
            for model in self._instances:
                if not model.is_busy:
                    await model.acquire()
                    return model
            
            # Create new instance if under limit
            if len(self._instances) < self._max_instances:
                instance_id = len(self._instances)
                new_model = self._model_class()
                
                try:
                    # Initialize with CUDA stream if available
                    if instance_id in self._cuda_streams:
                        with torch.cuda.stream(self._cuda_streams[instance_id]):
                            await new_model.setup(model_name)
                    else:
                        await new_model.setup(model_name)
                    
                    await new_model.acquire()
                    self._instances.append(new_model)
                    return new_model
                    
                except Exception as e:
                    logger.error(f"Failed to initialize model instance {instance_id}: {e}")
                    # Cleanup any partial initialization
                    if hasattr(new_model, '_model') and new_model._model is not None:
                        await self._cleanup_instance(new_model)
                    raise RuntimeError(f"Model initialization failed: {str(e)}")
            
            # All instances busy, wait for one
            logger.warning("All model instances busy. Waiting...")
            return await self._wait_for_available_model()

    async def _wait_for_available_model(self) -> TTSBaseModel:
        """
        Wait for an available model instance with configured timeout
        
        Returns:
            An available TTSBaseModel instance
        
        Raises:
            asyncio.TimeoutError if no model becomes available
        """
        try:
            return await asyncio.wait_for(
                self._find_available_model(),
                timeout=settings.model_request_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"No model instance available after {settings.model_request_timeout} seconds")
            raise

    async def _find_available_model(self) -> TTSBaseModel:
        """
        Continuously check for an available model instance
        
        Returns:
            First available TTSBaseModel instance
        """
        while True:
            async with self._instance_lock:
                for model in self._instances:
                    if not model.is_busy:
                        await model.acquire()
                        return model
            
            # Short sleep to prevent tight loop
            await asyncio.sleep(0.1)

    async def _cleanup_instance(self, model: TTSBaseModel):
        """
        Clean up resources for a model instance
        
        Args:
            model: The model instance to clean up
        """
        try:
            # Handle both ONNX and PyTorch models
            if hasattr(model, '_model'):
                if isinstance(model._model, torch.nn.Module):
                    model._model.cpu()
                    del model._model
                elif hasattr(model._model, 'release'):  # ONNX Runtime session
                    model._model.release()
            
            # Clear any cached data
            if hasattr(model, 'cache_clear'):
                model.cache_clear()
            
            # Force CUDA memory cleanup if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error cleaning up model instance: {e}")

    def release_model(self, model: TTSBaseModel):
        """
        Release a model instance back to the pool
        
        Args:
            model: The TTSBaseModel instance to release
        """
        model.release()

    async def shutdown(self):
        """
        Gracefully shutdown all model instances and cleanup resources
        """
        logger.info("Starting model manager shutdown...")
        
        async with self._instance_lock:
            # Clean up each instance
            for model in self._instances:
                try:
                    await self._cleanup_instance(model)
                except Exception as e:
                    logger.error(f"Error during instance cleanup: {e}")
            
            # Clear instance tracking
            self._instances.clear()
            
            # Clean up CUDA streams
            if torch.cuda.is_available():
                try:
                    for stream in self._cuda_streams.values():
                        stream.synchronize()
                        del stream
                    self._cuda_streams.clear()
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error cleaning up CUDA streams: {e}")
                    
        logger.info("Model manager shutdown complete")