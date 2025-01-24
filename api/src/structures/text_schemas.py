from pydantic import BaseModel, Field, field_validator
from typing import List, Union, Optional


class PhonemeRequest(BaseModel):
    text: str
    language: str = "a"  # Default to American English


class TextAlignment(BaseModel):
    """Character-level text alignment with timing"""
    characters: list[str]
    character_start_times_seconds: list[float]
    character_end_times_seconds: list[float]


class PhonemeAlignment(BaseModel):
    """Alignment between text and phonemes"""
    text: str
    phonemes: str
    start_idx: int
    end_idx: int

class PhonemeResponse(BaseModel):
    """Response from phonemize endpoint"""
    phonemes: str
    tokens: list[int]
    alignments: list[PhonemeAlignment]
    alignments: list[PhonemeAlignment]

class TimedResponse(BaseModel):
    """Response with audio and timing information"""
    audio: str
    input_text: str
    audio_duration: float
    alignment: TextAlignment
    phoneme_timing: list[dict]
    text_timing: list[list]


class StitchOptions(BaseModel):
    """Options for stitching audio chunks together"""
    gap_method: str = Field(
        default="static_trim",
        description="Method to handle gaps between chunks. Currently only 'static_trim' supported."
    )
    trim_ms: int = Field(
        default=0,
        ge=0,
        description="Milliseconds to trim from chunk boundaries when using static_trim"
    )

    @field_validator('gap_method')
    @classmethod
    def validate_gap_method(cls, v: str) -> str:
        if v != 'static_trim':
            raise ValueError("Currently only 'static_trim' gap method is supported")
        return v


class GenerateFromPhonemesRequest(BaseModel):
    phonemes: Union[str, List[str]] = Field(
        ...,
        description="Single phoneme string or list of phoneme chunks to stitch together"
    )
    voice: str = Field(..., description="Voice ID to use for generation")
    speed: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Speed factor for generation"
    )
    text: Optional[str] = Field(
        default=None,
        description="Original text for timing alignment. If provided, response will include text-level timing information."
    )
    language: str = Field(
        default="a",
        description="Language code for text phonemization ('a' for US English, 'b' for British English)"
    )
    options: Optional[StitchOptions] = Field(
        default=None,
        description="Optional settings for audio generation and stitching"
    )
