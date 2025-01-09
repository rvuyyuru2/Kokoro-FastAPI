import os
import numpy as np
import torch
import time
from loguru import logger
from models import build_model
from .text_processing import phonemize, tokenize

from .tts_base import TTSBaseModel
from ..core.config import settings
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

# @torch.no_grad()
# def forward(model, tokens, ref_s, speed):
#     """Forward pass through the model"""
#     device = ref_s.device
#     tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
#     input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
#     text_mask = length_to_mask(input_lengths).to(device)
#     bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
#     d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
#     s = ref_s[:, 128:]
#     d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
#     x, _ = model.predictor.lstm(d)
#     duration = model.predictor.duration_proj(x)
#     duration = torch.sigmoid(duration).sum(axis=-1) / speed
#     pred_dur = torch.round(duration).clamp(min=1).long()
#     pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
#     c_frame = 0
#     for i in range(pred_aln_trg.size(0)):
#         pred_aln_trg[i, c_frame : c_frame + pred_dur[0, i].item()] = 1
#         c_frame += pred_dur[0, i].item()
#     en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
#     F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
#     t_en = model.text_encoder(tokens, input_lengths, text_mask)
#     asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
#     return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()
@torch.no_grad()
def forward(model, tokens, ref_s, speed):
    """Forward pass through the model with light optimizations that preserve output quality"""
    device = ref_s.device
    
    # Keep original token handling but optimize device placement
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    
    # BERT and encoder pass
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    
    # Split reference signal once for efficiency
    s_content = ref_s[:, 128:]
    s_ref = ref_s[:, :128]
    
    # Predictor forward pass
    d = model.predictor.text_encoder(d_en, s_content, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    
    # Duration prediction - keeping original logic
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    
    # Alignment matrix construction - keeping original approach for quality
    pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item(), device=device)
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0, i].item()] = 1
        c_frame += pred_dur[0, i].item()
    
    # Matrix multiplications - reuse unsqueezed tensor
    pred_aln_trg = pred_aln_trg.unsqueeze(0)  # Do unsqueeze once
    en = d.transpose(-1, -2) @ pred_aln_trg
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s_content)
    
    # Text encoding and final decoding
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    
    return model.decoder(asr, F0_pred, N_pred, s_ref).squeeze().cpu().numpy()

# def length_to_mask(lengths):
#     """Create attention mask from lengths"""
#     mask = (
#         torch.arange(lengths.max())
#         .unsqueeze(0)
#         .expand(lengths.shape[0], -1)
#         .type_as(lengths)
#     )
#     mask = torch.gt(mask + 1, lengths.unsqueeze(1))
#     return mask

def length_to_mask(lengths):
    """Create attention mask from lengths - possibly optimized version"""
    max_len = lengths.max()
    # Create mask directly on the same device as lengths
    mask = torch.arange(max_len, device=lengths.device)[None, :].expand(lengths.shape[0], -1)
    # Avoid type_as by using the correct dtype from the start
    if lengths.dtype != mask.dtype:
        mask = mask.to(dtype=lengths.dtype)
    # Fuse operations  using broadcasting
    return mask + 1 > lengths[:, None]



class ModelInstance:
    """Wrapper for PyTorch model with CUDA stream management"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.lock = threading.Lock()
        
    def __call__(self, *args, **kwargs):
        """Thread-safe model inference with dedicated CUDA stream"""
        with self.lock:
            with torch.cuda.stream(self.stream):
                result = self.model(*args, **kwargs)
                # Ensure CUDA operations in this stream are completed
                self.stream.synchronize()
                return result

class TTSGPUModel(TTSBaseModel):
    _model_pool = Queue()
    _pool_size = settings.gpu_pool_size  # Use configured pool size or auto-scale
    _pool_lock = threading.Lock()
    _executor = None  # Will be initialized with correct pool size
    _device = "cuda"

    @classmethod
<<<<<<< Updated upstream
=======
    def get_instance(cls):
        """Get model instance from pool"""
        if cls._model_pool.empty():
            raise RuntimeError("Model pool not initialized. Call setup() first.")
        instance = cls._model_pool.get()
        cls._model_pool.put(instance)  # Put back for reuse
        return instance, cls.get_device()

    @classmethod
>>>>>>> Stashed changes
    def initialize(cls, model_dir: str, model_path: str):
        """Initialize pool of PyTorch models for GPU inference"""
        with cls._pool_lock:
            if not cls._model_pool.empty():
                # Get and return first instance (will be put back by get_instance)
                instance = cls._model_pool.get()
                cls._model_pool.put(instance)
                return instance

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            try:
                logger.info(f"Initializing pool of {cls._pool_size} GPU models")
                model_path = os.path.join(model_dir, settings.pytorch_model_path)
                
                # Calculate pool size if auto-scaling is enabled
                if cls._pool_size == 0:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                    # Convert GB to bytes (1GB = 1024^3 bytes)
                    memory_per_model = int(settings.gpu_memory_per_model_gb * (1024 ** 3))
                    cls._pool_size = max(1, int(free_memory / memory_per_model))
                    logger.info(f"Auto-scaled pool size to {cls._pool_size} based on {settings.gpu_memory_per_model_gb}GB per model")
                else:
                    logger.info(f"Using configured pool size of {cls._pool_size}")
                
                # Initialize thread pool with adjusted size
                cls._executor = ThreadPoolExecutor(max_workers=cls._pool_size)
                
                # Initialize all models in the pool
                for i in range(cls._pool_size):
                    # Initialize model with its own CUDA stream
                    model = build_model(model_path, cls._device)
                    instance = ModelInstance(model, cls._device)
                    cls._model_pool.put(instance)
                
                # Get and return first instance (will be put back by get_instance)
                instance = cls._model_pool.get()
                cls._model_pool.put(instance)
                return instance
                
            except Exception as e:
                logger.error(f"Failed to initialize GPU model pool: {e}")
                raise

    @classmethod
    def process_text(cls, text: str, language: str) -> tuple[str, list[int]]:
        """Process text into phonemes and tokens
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            tuple[str, list[int]]: Phonemes and token IDs
        """
        phonemes = phonemize(text, language)
        tokens = tokenize(phonemes)
        return phonemes, tokens

    @classmethod
    def generate_from_text(cls, text: str, voicepack: torch.Tensor, language: str, speed: float) -> tuple[np.ndarray, str]:
        """Generate audio from text
        
        Args:
            text: Input text
            voicepack: Voice tensor
            language: Language code
            speed: Speed factor
            
        Returns:
            tuple[np.ndarray, str]: Generated audio samples and phonemes
        """
        # Process text
        phonemes, tokens = cls.process_text(text, language)
        
        # Generate audio using thread-safe instance
        audio = cls.generate_from_tokens(tokens, voicepack, speed)
        
        return audio, phonemes

    @classmethod
    def generate_from_tokens(cls, tokens: list[int], voicepack: torch.Tensor, speed: float) -> np.ndarray:
        """Generate audio from tokens using thread-safe model instance
        
        Args:
            tokens: Token IDs
            voicepack: Voice tensor
            speed: Speed factor
            
        Returns:
            np.ndarray: Generated audio samples
        """
        model_instance, _ = cls.get_instance()
        
        # Get reference style
        ref_s = voicepack[len(tokens)]
        
        # Generate audio using thread-safe instance with dedicated CUDA stream
        audio = forward(model_instance.model, tokens, ref_s, speed)
            
        return audio
