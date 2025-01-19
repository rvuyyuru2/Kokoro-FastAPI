"""CPU-based PyTorch inference backend."""

import gc
import asyncio
from typing import Optional, List

import numpy as np
import torch
from loguru import logger

from ..builds.models import build_model
from ..core import paths
from ..structures.model_schemas import PyTorchCPUConfig
from .base import BaseModelBackend, ModelState


@torch.no_grad()
def forward(
    model: torch.nn.Module,
    tokens: list[int],
    ref_s: torch.Tensor,
    speed: float
) -> np.ndarray:
    """Forward pass through model.
    
    Args:
        model: PyTorch model
        tokens: Input tokens
        ref_s: Reference signal (shape: [1, n_features])
        speed: Speed multiplier
        
    Returns:
        Generated audio
    """
    device = ref_s.device
    
    # Initial tensor setup
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)

    # Split reference signals (style_dim=128 from config)
    style_dim = 128
    s_ref = ref_s[:, :style_dim].clone().to(device)
    s_content = ref_s[:, style_dim:].clone().to(device)

    # BERT and encoder pass
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    # Predictor forward pass
    d = model.predictor.text_encoder(d_en, s_content, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)

    # Duration prediction
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    del duration, x

    # Alignment matrix construction
    pred_aln_trg = torch.zeros(input_lengths.item(), pred_dur.sum().item(), device=device)
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0, i].item()] = 1
        c_frame += pred_dur[0, i].item()
    pred_aln_trg = pred_aln_trg.unsqueeze(0)

    # Matrix multiplications
    en = d.transpose(-1, -2) @ pred_aln_trg
    del d
    
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s_content)
    del en

    # Final text encoding and decoding
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    del t_en

    # Generate output
    output = model.decoder(asr, F0_pred, N_pred, s_ref)
    return output.squeeze().numpy()


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create attention mask from lengths."""
    max_len = lengths.max()
    mask = torch.arange(max_len, device=lengths.device)[None, :].expand(lengths.shape[0], -1)
    if lengths.dtype != mask.dtype:
        mask = mask.to(dtype=lengths.dtype)
    return mask + 1 > lengths[:, None]


class PyTorchCPUBackend(BaseModelBackend):
    """PyTorch CPU inference backend."""

    def __init__(self, config: Optional[PyTorchCPUConfig] = None):
        """Initialize CPU backend."""
        super().__init__()
        self._device = "cpu"
        self._model: Optional[torch.nn.Module] = None
        self._config = config or PyTorchCPUConfig()
        self._state = ModelState.UNINITIALIZED
        
        # Set thread counts
        torch.set_num_threads(self._config.num_threads)
        torch.set_num_interop_threads(self._config.inter_op_threads)

    async def load_model(self, path: str) -> None:
        """Load PyTorch model.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            
            logger.info(f"Loading PyTorch model: {model_path}")
            self._model = await build_model(model_path, self._device)
            self._state = ModelState.LOADED
            logger.info("PyTorch model loaded successfully")
            
        except Exception as e:
            self._state = ModelState.FAILED
            raise RuntimeError(f"Failed to load PyTorch model: {e}")

    async def generate(
        self,
        tokens: List[int],
        voice: torch.Tensor,
        speed: float = 1.0,
        stream: Optional[torch.cuda.Stream] = None  # Ignored for CPU backend
    ) -> np.ndarray:
        """Generate audio using CPU model.
        
        Args:
            tokens: Input token IDs
            voice: Voice embedding tensor
            speed: Speed multiplier
            stream: Ignored for CPU backend
            
        Returns:
            Generated audio samples
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_ready:
            raise RuntimeError("Model not ready for inference")

        try:
            # Check memory and cleanup if needed
            if self._check_memory():
                self._clear_memory()
                
            # Get reference style from voice pack
            ref_s = voice[len(tokens)].clone()
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)  # Add batch dimension if needed
            
            # Run inference in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                forward,
                self._model,
                tokens,
                ref_s,
                speed
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        return memory_gb > self._config.memory_threshold

    def _clear_memory(self) -> None:
        """Clear memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def warmup(self) -> None:
        """Run model warmup.
        
        Raises:
            RuntimeError: If warmup fails
        """
        if not self.is_loaded:
            raise RuntimeError("Cannot warmup - model not loaded")
            
        # Model warmup is handled by model manager
        self._state = ModelState.WARMED_UP
        logger.info("PyTorch model warmup completed")

    @property
    def state(self) -> ModelState:
        """Get current model state."""
        if self._model is None:
            return ModelState.UNINITIALIZED
        return self._state