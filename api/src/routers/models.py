from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Dict, List
import os
import aiofiles
import aiofiles.os
from loguru import logger

from ..services.tts_base import TTSBaseModel
from ..core.config import settings

router = APIRouter(prefix="/models", tags=["models"])

class ModelInfo(BaseModel):
    name: str
    path: str

@router.get("/list", response_model=List[str])
async def list_models():
    """List all available ONNX models"""
    models = []
    try:
        for file in os.listdir(settings.model_dir):
            if file.endswith('.onnx'):
                model_name = os.path.splitext(file)[0]
                models.append(model_name)
        logger.info(f"Found models: {models}")
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
    return models

@router.get("/current")
async def get_current_model():
    """Get the name of the currently loaded model"""
    current = TTSBaseModel.get_current_model()
    if not current:
        raise HTTPException(status_code=404, detail="No model currently loaded")
    return current

# @router.post("/add")
# async def add_model(file: UploadFile):
#     """Add a new ONNX model file
    
#     The file will be saved to the models directory and made available for use.
#     Only .onnx files are accepted.
#     """
#     if not file.filename.endswith('.onnx'):
#         raise HTTPException(status_code=400, detail="Only .onnx files are supported")
    
#     models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), settings.models_dir)
#     await aiofiles.os.makedirs(models_dir, exist_ok=True)
    
#     file_path = os.path.join(models_dir, file.filename)
    
#     try:
#         async with aiofiles.open(file_path, 'wb') as f:
#             content = await file.read()
#             await f.write(content)
            
#         # Reload models to include the new one
#         await TTSBaseModel.reload_models()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")
    
#     return {"message": f"Model {file.filename} added successfully"}
