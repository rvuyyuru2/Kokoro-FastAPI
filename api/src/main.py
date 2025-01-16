"""
FastAPI OpenAI Compatible API
"""

import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings
from .routers.development import router as dev_router
from .routers.openai_compatible import router as openai_router
from .services.tts_service import TTSService


def setup_logger():
    """Configure loguru logger with custom formatting"""
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
                "{level: <8} | "
                "{message}",
                "colorize": True,
                "level": "INFO",
            },
        ],
    }
    logger.remove()
    logger.configure(**config)
    logger.level("ERROR", color="<red>")


# Configure logger
setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for service initialization"""
    logger.info("Initializing TTS service...")

    # Initialize service and get a model to verify setup
    service = TTSService()
    model = await service.model_manager.get_model()
    try:
        boundary = "░" * 2*12
        startup_msg = f"""d

{boundary}

    ╔═╗┌─┐┌─┐┌┬┐
    ╠╣ ├─┤└─┐ │ 
    ╚  ┴ ┴└─┘ ┴ 
    ╦╔═┌─┐┬┌─┌─┐
    ╠╩╗│ │├┴┐│ │
    ╩ ╩└─┘┴ ┴└─┘

{boundary}
                """
        startup_msg += f"\nModel pool initialized on {model.get_device()}"
        startup_msg += f"\nMax concurrent instances: {settings.max_model_instances}"
        if settings.cuda_stream_per_instance and model.get_device() == "cuda":
            startup_msg += f"\nUsing separate CUDA streams per instance"
        startup_msg += f"\n{boundary}\n"
        logger.info(startup_msg)
    finally:
        service.model_manager.release_model(model)

    try:
        yield
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down TTS service...")
        await TTSService.shutdown_service()


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    openapi_url="/openapi.json",  # Explicitly enable OpenAPI schema
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(openai_router, prefix="/v1")
app.include_router(dev_router)  # New development endpoints


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
