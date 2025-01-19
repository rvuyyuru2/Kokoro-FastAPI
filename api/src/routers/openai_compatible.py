"""OpenAI-compatible TTS endpoints."""

from typing import Any, AsyncGenerator, List, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from ..services import get_service
from ..structures.schemas import OpenAISpeechRequest

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

async def get_tts_service():
    """Get TTS service instance."""
    return get_service()

async def process_voices(
    voice_input: Union[str, List[str]],
    service = Depends(get_tts_service)
) -> str:
    """Process voice input into a combined voice.
    
    Args:
        voice_input: Voice ID or list of voice IDs
        service: TTS service instance
        
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

    # Check if voices exist
    available_voices = await service.list_voices()
    for voice in voices:
        if voice not in available_voices:
            raise ValueError(
                f"Voice '{voice}' not found. Available: {', '.join(sorted(available_voices))}"
            )

    # Return single voice or combine
    return voices[0] if len(voices) == 1 else await service.combine_voices(voices)

async def stream_audio_chunks(
    request: OpenAISpeechRequest,
    client_request: Request,
    service = Depends(get_tts_service)
) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks.
    
    Args:
        request: Speech request
        client_request: FastAPI request
        service: TTS service instance
        
    Yields:
        Audio chunks
    """
    voice = await process_voices(request.voice, service=service)
    
    try:
        async for chunk in service.generate_stream(
            text=request.input,
            voice=voice,
            speed=request.speed,
            output_format=request.response_format
        ):
            # Check for client disconnect
            if await client_request.is_disconnected():
                logger.info("Client disconnected")
                break
            yield chunk
            
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise

@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    x_raw_response: str = Header(None, alias="x-raw-response"),
    service = Depends(get_tts_service)
):
    """OpenAI-compatible speech endpoint.
    
    Args:
        request: Speech request
        client_request: FastAPI request
        x_raw_response: Raw response header
        service: TTS service instance
        
    Returns:
        Audio response
    """
    try:
        voice = await process_voices(request.voice, service=service)
        content_type = CONTENT_TYPES.get(
            request.response_format,
            f"audio/{request.response_format}"
        )

        # Stream or complete file
        if request.stream:
            # Create generator with service instance
            async def stream_generator():
                async for chunk in stream_audio_chunks(request, client_request, service):
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
            audio_data = await service.generate_audio(
                text=request.input,
                voice=voice,
                speed=request.speed,
                output_format=request.response_format
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
async def list_voices(
    service = Depends(get_tts_service)
):
    """List available voices.
    
    Returns:
        List of voice IDs
    """
    try:
        voices = await service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voices/combine")
async def combine_voices(
    request: Union[str, List[str]],
    service = Depends(get_tts_service)
):
    """Combine multiple voices.
    
    Args:
        request: Voice IDs to combine
        service: TTS service instance
        
    Returns:
        Combined voice info
    """
    try:
        combined_voice = await process_voices(request, service=service)
        voices = await service.list_voices()
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
