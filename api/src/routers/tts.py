"""Core TTS endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from ..services import get_service

router = APIRouter(prefix="/tts", tags=["tts"])

# Content type mapping
CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "flac": "audio/flac"
}

async def get_tts_service():
    """Get TTS service instance."""
    return get_service()

@router.post("")
async def text_to_speech(
    text: str,
    voice: str,
    speed: float = 1.0,
    format: str = "wav",
    service = Depends(get_tts_service)
):
    """Generate complete audio file.
    
    Args:
        text: Input text
        voice: Voice ID
        speed: Speed multiplier
        format: Output format
        service: TTS service instance
        
    Returns:
        Audio file
    """
    try:
        audio_data = await service.generate_audio(
            text,
            voice,
            speed,
            format
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
    service = Depends(get_tts_service)
):
    """Generate streaming audio.
    
    Args:
        text: Input text
        voice: Voice ID
        speed: Speed multiplier
        format: Output format
        service: TTS service instance
        
    Returns:
        Audio stream
    """
    try:
        return StreamingResponse(
            service.generate_stream(text, voice, speed, format),
            media_type=CONTENT_TYPES.get(format, "application/octet-stream")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS streaming failed: {e}")
        raise HTTPException(status_code=500, detail="TTS streaming failed")

@router.post("/voices/combine")
async def combine_voices(
    voices: list[str],
    service = Depends(get_tts_service)
):
    """Combine multiple voices.
    
    Args:
        voices: List of voice IDs
        service: TTS service instance
        
    Returns:
        Combined voice ID
    """
    try:
        return await service.combine_voices(voices)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Voice combination failed: {e}")
        raise HTTPException(status_code=500, detail="Voice combination failed")

@router.get("/voices")
async def list_voices(
    service = Depends(get_tts_service)
):
    """List available voices.
    
    Returns:
        List of voice IDs
    """
    try:
        return await service.list_voices()
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to list voices")