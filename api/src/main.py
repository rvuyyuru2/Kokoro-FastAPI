"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings
from .routers import openai_compatible, tts
from .services import get_service

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tts.router)
app.include_router(openai_compatible.router)


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info("Initializing TTS service")
    
    # Initialize service
    service = get_service()
    
    # Log configuration
    logger.info(f"Using model backend: {service._model_manager.current_backend}")
    logger.info(f"Available voices: {len(await service.list_voices())}")
    logger.info("TTS service initialized")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down TTS service")
    
    # Get service instance
    service = get_service()
    
    # Clean up resources
    service._model_manager.unload_all()
    logger.info("Resources cleaned up")


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
