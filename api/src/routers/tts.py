"""Core TTS endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from ..core import paths
from ..utils.voice import combine_voices as combine_voice_tensors
from ..pipeline import get_factory
router = APIRouter(prefix="/tts", tags=["tts"])

# Content type mapping
CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "flac": "audio/flac"
}

async def get_pipeline_factory():
    """Get pipeline factory instance."""
    return await get_factory()

@router.post("")
async def text_to_speech(
    text: str,
    voice: str,
    speed: float = 1.0,
    format: str = "wav",
    factory = Depends(get_pipeline_factory)
):
    """Generate complete audio file.
    
    Args:
        text: Input text
        voice: Voice ID
        speed: Speed multiplier
        format: Output format
        factory: Pipeline factory instance
        
    Returns:
        Audio file
    """
    try:
        # Create whole file pipeline
        pipeline = await factory.create_pipeline("whole_file")
        
        # Generate audio
        audio_data = await pipeline.process(
            text=text,
            voice=voice,
            speed=speed,
            format=format,
            stream=False
        )
        
        return StreamingResponse(
            iter([audio_data]),
            media_type=CONTENT_TYPES.get(format, "application/octet-stream")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed")

@router.post("/stream")
async def stream_tts(
    text: str,
    voice: str,
    speed: float = 1.0,
    format: str = "wav",
    factory = Depends(get_pipeline_factory)
):
    """Generate streaming audio.
    
    Args:
        text: Input text
        voice: Voice ID
        speed: Speed multiplier
        format: Output format
        factory: Pipeline factory instance
        
    Returns:
        Audio stream
    """
    try:
        # Create streaming pipeline
        pipeline = await factory.create_pipeline("streaming")
        
        async def stream_generator():
            async for chunk in pipeline.process(
                text=text,
                voice=voice,
                speed=speed,
                format=format,
                stream=True
            ):
                yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type=CONTENT_TYPES.get(format, "application/octet-stream")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS streaming failed: {e}")
        raise HTTPException(status_code=500, detail="TTS streaming failed")

@router.post("/voices/combine")
async def combine_voices(voices: list[str]):
    """Combine multiple voices.
    
    Args:
        voices: List of voice IDs
        
    Returns:
        Combined voice ID
    """
    try:
        return await combine_voice_tensors(voices)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Voice combination failed: {e}")
        raise HTTPException(status_code=500, detail="Voice combination failed")

@router.get("/voices")
async def list_voices():
    """List available voices.
    
    Returns:
        List of voice IDs
    """
    try:
        return await paths.list_voices()
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to list voices")