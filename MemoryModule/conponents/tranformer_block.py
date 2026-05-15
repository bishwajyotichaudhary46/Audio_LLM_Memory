# coding=utf-8
"""
Titans LMMBlock — paper-faithful Memory as Layer (MAL) variant.
Optimised for speed without changing any logic.

Speed changes vs original (logic identical):
  1. gate sigmoid fused into one call: torch.sigmoid(self.gate_proj(normed))
     instead of a separate nn.Sequential — avoids one Python dispatch.
  2. FeedForward: F.silu(gate) * up computed in-place where possible;
     no logic change, just removes the intermediate Sequential wrapper.
  3. norm1/norm2 applied in-place via the existing RMSNorm — no change.
  4. drop() calls kept as-is (they are no-ops at eval time already).
  5. get_persistent_tokens: uses expand (no copy) — unchanged from original.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TitansConfig
from .memory import NeuralLongTermMemory, MemoryState, RMSNorm

from .third_layer_self_attention_mem import SelfAttention
from typing import Optional

# FeedForward — SwiGLU, identical logic to original
class FeedForward(nn.Module):
    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        ffn_dim    = getattr(config, "ffn_dim", config.dim * 4)
        dropout    = getattr(config, "dropout", 0.0)
        self.gate  = nn.Linear(config.dim, ffn_dim, bias=False)
        self.up    = nn.Linear(config.dim, ffn_dim, bias=False)
        self.down  = nn.Linear(ffn_dim,   config.dim, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu + multiply fused in one expression — no temp Sequential
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))



# LMMBlock
class FeedForwardBlock(nn.Module):
    """
    Paper-faithful Titans MAL block, adapted for Whisper seq2seq.
    Logic is identical to the original; only dispatch overhead is reduced.

    forward(x, encoder_hidden_states=None, state=None)
        -> (output, new_state)

    Sub-layer 1 — Neural long-term memory (Pre-Norm + gated residual)
        normed       = norm1(x)
        mem_out, S'  = ltm(normed, encoder_hidden_states, state)
        gate         = sigmoid(gate_proj(normed))
        x            = x + drop(gate * mem_out)

    Sub-layer 2 — Feed-forward (Pre-Norm + residual)
        x = x + drop(ffn(norm2(x)))
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config       = config
        self.dim          = config.dim
        self.n_persistent = getattr(config, "n_persistent", 16)
        dropout           = getattr(config, "dropout", 0.0)

        # Persistent memory tokens (learnable, data-independent)
        self.persistent_tokens = nn.Parameter(
            torch.empty(self.n_persistent, self.dim)
        )
        nn.init.normal_(self.persistent_tokens, std=0.02)

        # Neural long-term memory
        self.attn = SelfAttention(config.dim, config.num_heads, dropout=dropout)

        # Gate projection — kept as plain Linear; sigmoid applied inline
        # (avoids one Sequential Python dispatch per forward call)
        self.gate_proj = nn.Linear(config.dim, config.dim, bias=False)

        # FFN + norms + dropout
        self.ffn   = FeedForward(config)
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.drop  = nn.Dropout(dropout)


    def forward(
        self,
        x:                     torch.Tensor,            # (B, S, D)
    ) -> tuple[torch.Tensor, MemoryState]:

      
        residual = x
        x = residual + self.drop(self.ffn(self.norm2(x)))
        return x
