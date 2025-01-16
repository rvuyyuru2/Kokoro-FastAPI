from pathlib import Path
from typing import AsyncGenerator, List, Union
import aiofiles.os

from fastapi import APIRouter, Depends, Header, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from ..core.config import settings
from ..services.audio import AudioService
from ..services.tts_service import TTSService
from ..structures.schemas import OpenAISpeechRequest
from ..utils.paths import get_model_file, get_voice_file

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance with database session"""
    return TTSService()


async def process_voices(
    voice_input: Union[str, List[str]], tts_service: TTSService
) -> str:
    """Process voice input into a combined voice, handling both string and list formats"""
    # Convert input to list of voices
    if isinstance(voice_input, str):
        voices = [v.strip() for v in voice_input.split("+") if v.strip()]
    else:
        voices = voice_input

    if not voices:
        raise ValueError("No voices provided")

    # Check if all voices exist
    available_voices = await get_voice_file()  # Returns list of voice names
    for voice in voices:
        if voice not in available_voices:
            raise ValueError(
                f"Voice '{voice}' not found. Available voices: {', '.join(sorted(available_voices))}"
            )

    # If single voice, return it directly
    if len(voices) == 1:
        return voices[0]

    # Otherwise combine voices
    return await tts_service.combine_voices(voices=voices)


async def stream_audio_chunks(
    tts_service: TTSService, 
    request: OpenAISpeechRequest,
    client_request: Request
) -> AsyncGenerator[bytes, None]:
    """Stream audio chunks as they're generated with client disconnect handling"""
    try:
        # Validate model and voice before starting stream
        await tts_service._validate_model(request.model)
        voice_to_use = await process_voices(request.voice, tts_service)
        
        async for chunk in tts_service.generate_audio_stream(
            text=request.input,
            voice=voice_to_use,
            speed=request.speed,
            output_format=request.response_format,
            model=request.model,
        ):
            # Check if client is still connected
            if await client_request.is_disconnected():
                logger.info("Client disconnected, stopping audio generation")
                break
            yield chunk
            
    except RuntimeError as e:
        if "Model" in str(e):
            logger.error(f"Model error in audio streaming: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": "model_not_found",
                        "message": str(e),
                        "param": "model",
                        "type": "invalid_request_error"
                    }
                }
            )
        raise  # Re-raise other RuntimeErrors
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"Error in audio streaming: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
        )


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
    tts_service: TTSService = Depends(get_tts_service),
    x_raw_response: str = Header(None, alias="x-raw-response"),
):
    """OpenAI-compatible endpoint for text-to-speech"""
    try:
        # Validate model and voice before starting generation
        if request.model == "kokoro":
            # legacy v < 0.1.0
            request.model = settings.default_model

        await tts_service._validate_model(request.model)
        voice_to_use = await process_voices(request.voice, tts_service)

        # Set content type based on format
        content_type = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }.get(request.response_format, f"audio/{request.response_format}")

        # Check if streaming is requested (default for OpenAI client)
        if request.stream:
            # Stream audio chunks as they're generated
            return StreamingResponse(
                stream_audio_chunks(tts_service, request, client_request),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "X-Accel-Buffering": "no",  # Disable proxy buffering
                    "Cache-Control": "no-cache",  # Prevent caching
                    "Transfer-Encoding": "chunked",  # Enable chunked transfer encoding
                },
            )
        else:
            # Generate complete audio with specified model
            audio_bytes = await tts_service.generate_audio(
                text=request.input,
                voice=voice_to_use,
                speed=request.speed,
                output_format=request.response_format,
                model=request.model,
            )

            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",  # Prevent caching
                },
            )

    except RuntimeError as e:
        if "Model" in str(e):
            # Model-specific errors
            logger.error(f"Model error: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": "model_not_found",
                        "message": str(e),
                        "param": "model",
                        "type": "invalid_request_error"
                    }
                }
            )
        raise  # Re-raise other RuntimeErrors
    except ValueError as e:
        # Other validation errors
        logger.error(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error"
                }
            }
        )


@router.get("/audio/voices")
async def list_voices():
    """List all available voices for text-to-speech"""
    try:
        voices = await get_voice_file()  # Returns list of voice names
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
        )


@router.post("/audio/voices/combine")
async def combine_voices(
    request: Union[str, List[str]], tts_service: TTSService = Depends(get_tts_service)
):
    """Combine multiple voices into a new voice.

    Args:
        request: Either a string with voices separated by + (e.g. "voice1+voice2")
                or a list of voice names to combine

    Returns:
        Dict with combined voice name and list of all available voices

    Raises:
        HTTPException:
            - 400: Invalid request (wrong number of voices, voice not found)
            - 500: Server error (file system issues, combination failed)
    """
    try:
        combined_voice = await process_voices(request, tts_service)
        voices = await get_voice_file()  # Returns list of voice names
        return {"voices": voices, "voice": combined_voice}

    except ValueError as e:
        logger.error(f"Invalid voice combination request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error"
                }
            }
        )

    except Exception as e:
        logger.error(f"Server error during voice combination: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error"
                }
            }
        )


@router.get("/models")
async def list_models():
    """List all available models in OpenAI format"""
    try:
        models = []
        # Scan both model directories
        for path in [Path(settings.model_dir)]:
            if not path.exists():
                continue
                
            iterfiles = await aiofiles.os.scandir(path)
            for entry in iterfiles:
                if not entry.is_file():
                    continue
                    
                if entry.name.endswith('.onnx') or entry.name.endswith('.pth'):
                    model_path = Path(entry.path)
                    stat = await aiofiles.os.stat(model_path)
                    model_type = 'onnx' if model_path.suffix == '.onnx' else 'pth'
                    
                    models.append({
                        "id": model_path.stem,  # Keep full name including version
                        "object": "model",
                        "created": int(stat.st_ctime),
                        "owned_by": model_type
                    })
                    
        return {
            "object": "list",
            "data": sorted(models, key=lambda x: x["id"])
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
        )
