"""Main FastAPI application."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core import paths
from .core.config import settings
from .routers import openai_compatible, tts
from .services import get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for service initialization."""
    logger.info("Initializing TTS service...")

    # Initialize plugin manager first
    from .plugins.hooks import initialize_plugin_manager, get_plugin_manager
    try:
        plugin_manager = get_plugin_manager()
    except RuntimeError:
        plugin_manager = initialize_plugin_manager()
    logger.info("Plugin manager initialized")

    # Initialize service
    service = get_service()
    model_manager = service._model_manager

    try:
        # Initialize model
        model_path = (
            settings.pytorch_model_path
            if not settings.use_onnx
            else settings.onnx_model_path
        )
        await model_manager.load_model(model_path)  # This includes warmup
        service._validate_model()  # Ensure model is properly loaded

        # Run warmup with first available voice
        voice_names = await service.list_voices()
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
            
            async for _ in service.generate_stream(sample_text, first_voice):
                pass
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
        # Use service's shutdown method for proper cleanup
        await service.shutdown()


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
