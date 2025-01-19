"""Main FastAPI application."""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core import paths
from .core.config import settings
from .routers import openai_compatible, tts
from .inference import get_manager
from .pipeline import get_factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for initialization."""
    logger.info("Initializing TTS system...")

    # Initialize plugin manager first
    from .plugins.hooks import initialize_plugin_manager, get_plugin_manager
    try:
        plugin_manager = get_plugin_manager()
    except RuntimeError:
        plugin_manager = initialize_plugin_manager()
    logger.info("Plugin manager initialized")

    # Initialize model manager
    model_manager = get_manager()

    try:
        # Initialize model
        model_path = (
            settings.pytorch_model_path
            if not settings.use_onnx
            else settings.onnx_model_path
        )
        # Load and warmup model
        await model_manager.load_model(model_path)
        backend = model_manager.get_backend()
        await backend.warmup()
        
        if not backend.is_loaded:
            raise RuntimeError("Model failed to load")

        # Initialize pipeline factory
        factory = await get_factory()

        # Run voice warmup
        voice_names = await paths.list_voices()
        if not voice_names:
            logger.error("No voices available for warmup")
            raise RuntimeError("No voices found")
            
        first_voice = voice_names[0]
        logger.info(f"Running warmup inference with voice: {first_voice}")
        
        # Run warmup inference with voice
        try:
            sample_text = (await paths.read_file(
                os.path.join(os.path.dirname(__file__), "core", "don_quixote.txt")
            )).splitlines()[0]
            
            # Create whole file pipeline for warmup
            pipeline = await factory.create_pipeline("whole_file")
            await pipeline.process(
                text=sample_text,
                voice=first_voice,
                stream=False  # Get complete audio data
            )
            logger.info("Warmup complete")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            raise RuntimeError(f"Warmup failed: {e}")

        # Log startup info
        boundary = "░" * 24
        startup_msg = f"""
{boundary}

    ╔═╗┌─┐┌─┐┌┬┐
    ╠╣ ├─┤└─┐ │
    ╚  ┴ ┴└─┘ ┴
    ╦╔═┌─┐┬┌─┌─┐
    ╠╩╗│ │├┴┐│ │
    ╩ ╩└─┘┴ ┴└─┘

{boundary}
        """
        startup_msg += f"\nModel backend: {model_manager.current_backend}"
        startup_msg += f"\nAvailable voices: {len(voice_names)}"
        startup_msg += f"\n{boundary}\n"
        logger.info(startup_msg)

        yield

    finally:
        # Cleanup resources
        model_manager.unload_all()


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tts.router)
app.include_router(openai_compatible.router)


@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "version": app.version
    }


@app.get("/")
async def root():
    """Root endpoint.
    
    Returns:
        API information
    """
    return {
        "name": app.title,
        "description": app.description,
        "version": app.version,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
