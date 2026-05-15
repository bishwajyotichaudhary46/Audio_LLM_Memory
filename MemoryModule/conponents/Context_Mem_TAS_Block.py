# coding=utf-8
"""
Titans ContextMemTASBlock — paper-faithful Memory as Layer (MAL) variant.
Optimised for speed without changing any logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TitansConfig
from .memory import RMSNorm
from .TAS_NN import TASNeuralLongTermMemory, TASState   # TASState replaces MemoryState


# ── FeedForward ───────────────────────────────────────────────────────────────

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
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


# ── ContextMemTASBlock ────────────────────────────────────────────────────────

class ContextMemTASBlock(nn.Module):
    """
    Titans MAL block with Trajectory-Aware Surprise.

    forward(x, encoder_hidden_states=None, state=None)
        -> (output, new_state: TASState)

    Sub-layer 1 — Neural long-term memory (Pre-Norm + gated residual)
        normed          = norm1(x)
        mem_out, S'     = ltm(normed, encoder_hidden_states, state)
        gate            = sigmoid(gate_proj(normed))
        x               = x + drop(gate * mem_out)

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

        # Neural long-term memory — now TAS variant
        self.ltm = TASNeuralLongTermMemory(config)

        # Gate projection
        self.gate_proj = nn.Linear(config.dim, config.dim, bias=False)

        # FFN + norms + dropout
        self.ffn   = FeedForward(config)
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.drop  = nn.Dropout(dropout)

    def get_persistent_tokens(self, batch_size: int) -> torch.Tensor:
        """(B, n_persistent, D) — expand, no copy."""
        return self.persistent_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def init_state(self, device: torch.device) -> TASState:
        """
        Convenience: initialise a fresh TASState for this block.
        Callers (e.g. the model loop) should call this once per sequence.
        """
        return self.ltm.init_state(device)

    def forward(
        self,
        x:                     torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        state:                 TASState | None      = None,   # ← was MemoryState
    ) -> tuple[torch.Tensor, TASState]:                       # ← was MemoryState

        # ── Sub-layer 1: long-term memory ─────────────────────────────
        residual = x
        normed   = self.norm1(x)

        mem_out, new_state = self.ltm(
            normed,
            encoder_hidden_states=encoder_hidden_states,
            state=state,   # TASNeuralLongTermMemory.init_state called internally
        )                  # if state is None

        gate = torch.sigmoid(self.gate_proj(normed))
        x    = residual + self.drop(gate * mem_out)

        # ── Sub-layer 2: feed-forward ─────────────────────────────────
        residual = x
        x        = residual + self.drop(self.ffn(self.norm2(x)))

        return x, new_state
