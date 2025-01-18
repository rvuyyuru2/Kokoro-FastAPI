"""GPU-based PyTorch inference backend."""

import gc
import os
from typing import Optional

import numpy as np
import torch
from loguru import logger

from ..builds.models import build_model
from ..structures.model_schemas import GPUConfig
from .base import BaseModelBackend


@torch.no_grad()
def forward(model: torch.nn.Module, tokens: list[int], ref_s: torch.Tensor, speed: float) -> np.ndarray:
    """Forward pass through model with memory management.
    
    Args:
        model: PyTorch model
        tokens: Input tokens
        ref_s: Reference signal
        speed: Speed multiplier
        
    Returns:
        Generated audio
    """
    device = ref_s.device
    
    try:
        # Initial tensor setup
        tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        # Split reference signals
        s_content = ref_s[:, 128:].clone().to(device)
        s_ref = ref_s[:, :128].clone().to(device)

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
        del duration, x  # Free large intermediates

        # Alignment matrix construction
        pred_aln_trg = torch.zeros(
            input_lengths.item(),
            pred_dur.sum().item(),
            device=device
        )
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + pred_dur[0, i].item()] = 1
            c_frame += pred_dur[0, i].item()
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        # Matrix multiplications with cleanup
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
        result = output.squeeze().cpu().numpy()
        
        return result
        
    finally:
        # Clean up largest tensors
        del pred_aln_trg, asr


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create attention mask from lengths.
    
    Args:
        lengths: Sequence lengths
        
    Returns:
        Boolean mask tensor
    """
    max_len = lengths.max()
    mask = torch.arange(max_len, device=lengths.device)[None, :].expand(
        lengths.shape[0], -1
    )
    if lengths.dtype != mask.dtype:
        mask = mask.to(dtype=lengths.dtype)
    return mask + 1 > lengths[:, None]


class GPUBackend(BaseModelBackend):
    """PyTorch GPU inference backend."""

    def __init__(self):
        """Initialize GPU backend."""
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self._device = "cuda"
        self._model: Optional[torch.nn.Module] = None
        self._config = GPUConfig()

    def load_model(self, path: str) -> None:
        """Load PyTorch model.
        
        Args:
            path: Path to model file
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            if not os.path.exists(path):
                raise RuntimeError(f"Model not found: {path}")

            logger.info(f"Loading PyTorch model: {path}")
            self._model = build_model(path, self._device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")

    def generate(
        self,
        tokens: list[int],
        voice: torch.Tensor,
        speed: float = 1.0
    ) -> np.ndarray:
        """Generate audio using GPU model.
        
        Args:
            tokens: Input token IDs
            voice: Voice embedding tensor
            speed: Speed multiplier
            
        Returns:
            Generated audio samples
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # Check memory pressure
            if self._check_memory():
                self._clear_memory()

            # Prepare input
            ref_s = voice[len(tokens)].clone().to(self._device)
            
            # Generate audio
            return forward(self._model, tokens, ref_s, speed)
            
        except RuntimeError as e:
            if "out of memory" in str(e) and self._config.retry_on_oom:
                logger.warning("OOM detected, attempting recovery")
                self._clear_memory(full=True)
                
                # Retry generation
                ref_s = voice[len(tokens)].clone().to(self._device)
                return forward(self._model, tokens, ref_s, speed)
            raise
            
        finally:
            if self._config.sync_cuda:
                torch.cuda.synchronize()

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold.
        
        Returns:
            True if memory should be cleared
        """
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1e9
            return memory_gb > self._config.memory_threshold
        return False

    def _clear_memory(self, full: bool = False) -> None:
        """Clear GPU memory.
        
        Args:
            full: Whether to perform full cleanup
        """
        if torch.cuda.is_available():
            if full:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if full:
                gc.collect()
                
            # Log memory stats
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(
                f"GPU memory after cleanup: "
                f"Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB"
            )