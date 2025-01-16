from typing import Any, Dict, List
import torch
from pathlib import Path

from ..style_interface import StyleStrategy
from ...service_interfaces import TTSServiceProtocol
from ....core.config import settings


class CombineVoicesStrategy(StyleStrategy):
    """Strategy for combining multiple voices into a new style"""
    
    def __init__(self, tts_service: TTSServiceProtocol):
        self.tts_service = tts_service
        
    async def generate_style(self, **kwargs) -> torch.Tensor:
        """Generate a combined voice style from multiple voices
        
        Args:
            voices (List[str]): List of voice names to combine
            output_name (str, optional): Name for the combined voice file
            
        Returns:
            torch.Tensor: The combined voice tensor
        """
        voices = kwargs.get('voices', [])
        output_name = kwargs.get('output_name')
        
        if len(voices) < 2:
            raise ValueError("At least 2 voices are required for combination")
            
        # Load voices
        t_voices: List[torch.Tensor] = []
        v_name: List[str] = []
        
        for voice in voices:
            voice_path = await self.tts_service._find_voice(voice)
            if not voice_path:
                raise ValueError(f"Voice not found: {voice}")
                
            model = await self.tts_service.model_manager.get_model()
            try:
                voicepack = torch.load(
                    str(voice_path), 
                    map_location=model.get_device(), 
                    weights_only=True
                )
            finally:
                self.tts_service.model_manager.release_model(model)
                
            t_voices.append(voicepack)
            v_name.append(voice)
            
        # Combine voices
        combined = torch.mean(torch.stack(t_voices), dim=0)
        
        # Save if output name provided
        if output_name:
            save_path = Path(settings.voices_dir) / f"{output_name}.pt"
            torch.save(combined, str(save_path))
            
        return combined
        
    def get_required_params(self) -> Dict[str, Any]:
        """Get required parameters for voice combination
        
        Returns:
            Dict containing parameter names and their types
        """
        return {
            'voices': List[str],
            'output_name': str  # Optional
        }
        
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters for voice combination
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            bool: True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid
        """
        if 'voices' not in params:
            raise ValueError("'voices' parameter is required")
            
        voices = params.get('voices', [])
        if not isinstance(voices, list):
            raise ValueError("'voices' must be a list")
            
        if len(voices) < 2:
            raise ValueError("At least 2 voices are required")
            
        for voice in voices:
            if not isinstance(voice, str):
                raise ValueError("All voice names must be strings")
                
        output_name = params.get('output_name')
        if output_name is not None and not isinstance(output_name, str):
            raise ValueError("'output_name' must be a string if provided")
            
        return True