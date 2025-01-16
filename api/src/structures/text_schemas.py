from pydantic import BaseModel, Field


class PhonemeRequest(BaseModel):
    text: str
    language: str = "a"  # Default to American English


class PhonemeResponse(BaseModel):
    phonemes: str
    tokens: list[int]


class GenerateFromPhonemesRequest(BaseModel):
    phonemes: str
    voice: str = Field(..., description="Voice ID to use for generation")
    speed: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Speed factor for generation"
    )
    model: str = Field(
        default=None, description="Model ID to use for generation. If not specified, uses default model."
    )
