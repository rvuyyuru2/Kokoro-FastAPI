"""OpenAI-compatible TTS endpoints."""

from typing import Any, AsyncGenerator, List, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from ..core import paths
from ..pipeline import get_factory
from ..structures.schemas import OpenAISpeechRequest
from ..utils.voice import combine_voices as combine_voice_tensors

router = APIRouter(
    prefix="/v1",  # Changed to /v1 for OpenAI compatibility
    tags=["OpenAI Compatible TTS"]
)

# Content type mapping
CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm"
}

async def get_pipeline_factory():
    """Get pipeline factory instance."""
    return await get_factory()

async def validate_voice(voice: str) -> str:
    """Validate voice exists.
    
    Args:
        voice: Voice ID
        
    Returns:
        Validated voice ID
        
    Raises:
        ValueError: If voice not found
    """
    available_voices = await paths.list_voices()
    if voice not in available_voices:
        raise ValueError(
            f"Voice '{voice}' not found. Available: {', '.join(sorted(available_voices))}"
        )
    return voice

async def process_voices(voice_input: Union[str, List[str]]) -> str:
    """Process voice input into a combined voice.
    
    Args:
        voice_input: Voice ID or list of voice IDs
        
    Returns:
        Combined voice ID
    """
    # Convert input to list
    if isinstance(voice_input, str):
        voices = [v.strip() for v in voice_input.split("+") if v.strip()]
    else:
        voices = voice_input

    if not voices:
        raise ValueError("No voices provided")

    # Validate all voices
    for voice in voices:
        await validate_voice(voice)

    # Return single voice or combine
    return voices[0] if len(voices) == 1 else await combine_voice_tensors(voices)

@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
    factory = Depends(get_pipeline_factory)
):
    """OpenAI-compatible speech endpoint.
    
    Args:
        request: Speech request
        client_request: FastAPI request
        x_raw_response: Raw response header
        factory: Pipeline factory instance
        
    Returns:
        Audio response
    """
    try:
        voice = await process_voices(request.voice)
        content_type = CONTENT_TYPES.get(
            request.response_format,
            f"audio/{request.response_format}"
        )

        # Create appropriate pipeline
        pipeline = await factory.create_pipeline(
            "streaming" if request.stream else "whole_file"
        )

        if request.stream:
            # Create streaming response
            async def stream_generator():
                # Don't await the process call - it returns an async generator
                async for chunk in pipeline.process(
                    text=request.input,
                    voice=voice,
                    speed=request.speed,
                    format=request.response_format,
                    stream=True
                ):
                    if await client_request.is_disconnected():
                        logger.info("Client disconnected")
                        break
                    yield chunk

            return StreamingResponse(
                stream_generator(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            # Generate complete file
            audio_data = await pipeline.process(
                text=request.input,
                voice=voice,
                speed=request.speed,
                format=request.response_format,
                stream=False
            )
            
            return Response(
                content=audio_data,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache"
                }
            )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Server error", "message": str(e)}
        )

@router.get("/voices")
async def list_voices():
    """List available voices.
    
    Returns:
        List of voice IDs
    """
    try:
        voices = await paths.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voices/combine")
async def combine_voices_endpoint(request: Union[str, List[str]]):
    """Combine multiple voices.
    
    Args:
        request: Voice IDs to combine
        
    Returns:
        Combined voice info
    """
    try:
        combined_voice = await process_voices(request)
        voices = await paths.list_voices()
        return {
            "voices": voices,
            "voice": combined_voice
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Voice combination failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Server error", "message": str(e)}
        )
