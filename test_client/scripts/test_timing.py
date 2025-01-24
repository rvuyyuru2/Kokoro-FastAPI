"""Test script for text and phoneme timing features."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from loguru import logger

# Configure debug logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")


class TimingTest:
    """Test text and phoneme timing features."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize test client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path("output/timing_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def test_timing(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        save_audio: bool = True
    ) -> Dict:
        """Test timing feature with text input.
        
        Args:
            text: Input text
            voice: Voice ID to use
            speed: Speed factor
            save_audio: Whether to save generated audio
            
        Returns:
            Dict containing test results and timing information
        """
        logger.info(f"Testing timing with text: {text}")
        
        # First get phonemes
        async with httpx.AsyncClient() as client:
            phoneme_response = await client.post(
                f"{self.base_url}/subtitled/phonemize",
                json={"text": text, "language": "a"}
            )
            phoneme_response.raise_for_status()
            phonemes = phoneme_response.json()["phonemes"]
            
            # Generate audio with timing
            response = await client.post(
                f"{self.base_url}/subtitled/generate_from_phonemes",
                json={
                    "text": text,
                    "phonemes": phonemes,
                    "voice": voice,
                    "speed": speed,
                    "language": "a"  # Original text already included above
                }
            )
            response.raise_for_status()
            
            # Parse response JSON
            response_data = response.json()
            
            # Save results
            result = {
                "text": text,
                "phonemes": phonemes,
                "voice": voice,
                "speed": speed,
                "phoneme_timing": response_data["phoneme_timing"],
                "text_timing": response_data["text_timing"]
            }
            all_else = {k: v for k, v in response_data.items() if k not in result and k != "audio"}
            print(json.dumps(all_else, indent=2))
            
            # Save audio if requested
            if save_audio:
                audio_path = self.output_dir / f"{voice}_{speed}.wav"
                # Decode base64 audio
                import base64
                audio_bytes = base64.b64decode(response_data["audio"])
                audio_path.write_bytes(audio_bytes)
                result["audio_path"] = str(audio_path)
                
            # Save timing info
            timing_path = self.output_dir / f"{voice}_{speed}_timing.json"
            timing_path.write_text(json.dumps(result, indent=2))
            
            return result

async def main(url: str = "http://localhost:8000", voice: str = "af_bella", speed: float = 1.0, text: str = None, **kwargs):
    """Run timing tests.
    
    Args:
        url: Base URL of the API server
        voice: Voice ID to use
        speed: Speed factor
        text: Optional text to test with
    """
    test = TimingTest(base_url=url)
    
    # Use provided text or default test text
    test_text = text or "The quick brown fox jumps over the lazy dog."
    
    # Run single test with provided parameters
    result = await test.test_timing(test_text, voice=voice, speed=speed)
    # Use ensure_ascii=False to properly display Unicode characters
    logger.info(f"Test result: {json.dumps(result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())