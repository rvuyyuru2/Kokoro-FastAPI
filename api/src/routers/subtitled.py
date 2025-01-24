from typing import List, Dict, Tuple

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger

from ..services.audio import AudioService
from ..services.text_processing import phonemize, tokenize
from ..services.tts_service import TTSService
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
    TextAlignment,
    TimedResponse,
    PhonemeAlignment
)
from ..services.text_processing.phonemizer import PhonemeAlignment as PhonemizerAlignment

router = APIRouter(tags=["text processing"])


async def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance"""
    return await TTSService.create()  # Create service with properly initialized managers


@router.post("/text/phonemize", response_model=PhonemeResponse, tags=["deprecated"])
@router.post("/subtitled/phonemize", response_model=PhonemeResponse)
async def phonemize_text(request: PhonemeRequest) -> PhonemeResponse:
    """Convert text to phonemes and tokens

    Args:
        request: Request containing text and language
        tts_service: Injected TTSService instance

    Returns:
        Phonemes and token IDs
    """
    try:
        if not request.text:
            raise ValueError("Text cannot be empty")

        # Get phonemes with alignments
        phonemes, alignments = phonemize(request.text, request.language)
        if not phonemes:
            raise ValueError("Failed to generate phonemes")

        # Keep original IPA phonemes for response
        original_phonemes = phonemes

        # Get tokens (without adding start/end tokens to match process_text behavior)
        tokens = tokenize(phonemes)
        
        # Convert alignments to Pydantic models
        alignment_models = [
            PhonemeAlignment(
                text=align.text,
                phonemes=align.phonemes,
                start_idx=align.start_idx,
                end_idx=align.end_idx
            ) for align in alignments
        ]
        
        return PhonemeResponse(
            phonemes=original_phonemes,  # Use original IPA phonemes
            tokens=tokens,
            alignments=alignment_models
        )
    except ValueError as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )


@router.post("/text/generate_from_phonemes", tags=["deprecated"])
@router.post("/subtitled/generate_from_phonemes")
async def generate_from_phonemes(
    request: GenerateFromPhonemesRequest,
    tts_service: TTSService = Depends(get_tts_service),
) -> Response:
    """Generate audio directly from phonemes with timing information

    Args:
        request: Request containing phonemes and generation parameters
        tts_service: Injected TTSService instance

    Returns:
        WAV audio bytes with timing metadata in headers
    """
    # Validate phonemes first
    if not request.phonemes:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request", "message": "Phonemes cannot be empty"},
        )

    try:
        # Validate voice exists
        available_voices = await tts_service.list_voices()
        if request.voice not in available_voices:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid request",
                    "message": f"Voice not found: {request.voice}",
                },
            )

        # Get text alignments if original text was provided
        text_alignments = []
        if request.text:  # GenerateFromPhonemesRequest should have text field
            logger.debug(f"Getting alignments for text: {request.text}")
            phonemes, text_alignments = phonemize(request.text, request.language)
            logger.debug(f"Got phonemes: {phonemes}")
            logger.debug(f"Got {len(text_alignments)} alignments:")
            for align in text_alignments:
                logger.debug(f"  text='{align.text}' phonemes='{align.phonemes}' start={align.start_idx} end={align.end_idx}")

        # Handle both single string and list of chunks, keeping original IPA
        if isinstance(request.phonemes, str):
            phoneme_chunks = [request.phonemes]
        else:
            phoneme_chunks = request.phonemes
        audio_chunks = []
        timing_info = []

        # Load voice tensor first since we'll need it for all chunks
        voice_tensor = await tts_service._voice_manager.load_voice(
            request.voice,
            device=tts_service.model_manager.get_backend().device
        )

        try:
            # Process each chunk
            for chunk in phoneme_chunks:
                # Convert chunk to tokens
                tokens = tokenize(chunk)
                tokens = [0] + tokens + [0]  # Add start/end tokens

                # Validate chunk length
                if len(tokens) > 510:  # 510 to leave room for start/end tokens
                    raise ValueError(
                        f"Chunk too long ({len(tokens)} tokens). Each chunk must be under 510 tokens."
                    )

                # Generate audio for chunk
                audio, pred_dur = await tts_service.model_manager.generate(
                    tokens,
                    voice_tensor,
                    speed=request.speed
                )
                if audio is not None:
                    audio_chunks.append(audio)

                    # Convert frame durations to timestamps
                    frame_rate = 24000  # Audio sample rate
                    hop_length = 240    # Frames per hop (for 24kHz)
                    
                    # Calculate cumulative durations
                    cum_dur = np.cumsum(pred_dur)
                    timestamps = (cum_dur * hop_length / frame_rate).tolist()
                    
                    # Create timing info with original IPA phonemes
                    chunk_timing = {
                        'phonemes': phonemes,  # Use original IPA phonemes from phonemize()
                        'timestamps': timestamps
                    }
                    
                    # Add text timing if available
                    if text_alignments:
                        # Map phoneme timings to text positions
                        text_timings = []
                        for align in text_alignments:
                            # Use exact character positions from alignment
                            text = align.text.strip()
                            text_len = len(text)
                            
                            # Get timing boundaries using alignment positions
                            start_time = timestamps[align.start_idx] if align.start_idx < len(timestamps) else timestamps[-1]
                            end_time = timestamps[min(align.end_idx, len(timestamps)-1)]
                            total_duration = end_time - start_time
                            
                            logger.debug(f"Processing alignment: text='{text}' phonemes='{align.phonemes}' at positions {align.start_idx}-{align.end_idx}")
                            logger.debug(f"Mapped to timing {start_time:.3f} -> {end_time:.3f}")
                            
                            # Create arrays for character timing
                            characters = list(text)
                            char_start_times = []
                            char_end_times = []
                            
                            # Calculate timing for each character using exact character positions
                            for i, char in enumerate(characters):
                                # Get character position relative to word boundaries
                                char_pos = align.start_idx + i
                                char_proportion = (char_pos - align.start_idx) / (align.end_idx - align.start_idx)
                                next_proportion = (char_pos + 1 - align.start_idx) / (align.end_idx - align.start_idx)
                                
                                # Calculate precise timing using phoneme boundaries and character position
                                char_start = start_time + (total_duration * char_proportion)
                                char_end = start_time + (total_duration * next_proportion)
                                
                                # Ensure exact boundaries for first and last characters
                                if i == 0:
                                    char_start = start_time
                                if i == text_len - 1:
                                    char_end = end_time
                                    
                                # Round to 3 decimal places
                                char_start_times.append(round(char_start, 3))
                                char_end_times.append(round(char_end, 3))
                                    
                            logger.debug(f"Aligned text '{text}' ({len(characters)} chars) with phonemes '{align.phonemes}' from {start_time} to {end_time}")
                            
                            text_timings.append({
                                'text': align.text,
                                'start': start_time,
                                'end': end_time,
                                'alignment': {
                                    'characters': characters,
                                    'character_start_times_seconds': char_start_times,
                                    'character_end_times_seconds': char_end_times
                                }
                            })
                        chunk_timing['text_timing'] = text_timings
                        
                    timing_info.append(chunk_timing)

            # Combine chunks if needed
            if len(audio_chunks) > 1:
                audio = np.concatenate(audio_chunks)
            elif len(audio_chunks) == 1:
                audio = audio_chunks[0]
            else:
                raise ValueError("No audio chunks were generated")

        finally:
            # Clean up voice tensor
            del voice_tensor
            torch.cuda.empty_cache()

        # Create response with both audio and timing data
        import base64
        from fastapi.responses import JSONResponse

        # Convert audio to base64
        wav_bytes = AudioService.convert_audio(
            audio, 24000, "wav", is_first_chunk=True, is_last_chunk=True, stream=False,
        )
        audio_b64 = base64.b64encode(wav_bytes).decode()

        # Calculate audio duration
        audio_duration = len(audio) / 24000  # Sample rate is 24000

        # Combine all character alignments into a single alignment
        all_characters = []
        all_start_times = []
        all_end_times = []
        
        if timing_info:
            for chunk in timing_info:
                for timing in chunk.get('text_timing', []):
                    if 'alignment' in timing:
                        align = timing['alignment']
                        all_characters.extend(align['characters'])
                        all_start_times.extend(align['character_start_times_seconds'])
                        all_end_times.extend(align['character_end_times_seconds'])
        
        # Create final alignment
        alignment = TextAlignment(
            characters=all_characters,
            character_start_times_seconds=all_start_times,
            character_end_times_seconds=all_end_times
        )

        # Prepare response data using TimedResponse schema
        response_data = TimedResponse(
            audio=audio_b64,
            audio_duration=audio_duration,
            input_text=request.text or "",
            alignment=alignment,
            phoneme_timing=timing_info,
            text_timing=[chunk.get('text_timing', []) for chunk in timing_info] if timing_info else []
        ).dict()

        return JSONResponse(content=response_data)

    except ValueError as e:
        logger.error(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400, detail={"error": "Invalid request", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
