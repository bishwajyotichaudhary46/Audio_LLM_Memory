# coding=utf-8
"""
TitansConfig — configuration dataclass for all Titans modules.

New fields vs original:
  chunk_size       : tokens per chunk for parallel update (Section 3.2)
  n_persistent     : number of persistent memory tokens  (Section 3.3)
  memory_lr        : scale for theta gate (per-token learning rate)
  memory_momentum  : scale for eta gate   (per-token momentum coeff)
  num_memory_layers: depth of MemoryMLP
  memory_hidden_dim: hidden dim of MemoryMLP (defaults to dim*2)
  ffn_dim          : FFN expansion dim in LMMBlock (defaults to dim*4)
  dropout          : dropout probability
  activation       : nonlinearity for MemoryMLP ("silu" | "gelu" | "relu")
  init_std         : weight init std
"""

from dataclasses import dataclass, field


@dataclass
class TitansConfig:
    # Core dimensions
    dim:               int   = 512
    num_heads:         int   = 8
    num_layers:        int   = 4
    vocab_size:        int   = 51865    # Whisper default

    # Chunked parallel update (Section 3.2)
    chunk_size:        int   = 64       # tokens processed in parallel per chunk

    # Persistent memory tokens (Section 3.3)
    n_persistent:      int   = 16       # number of learnable persistent tokens

    # Memory MLP depth and width
    num_memory_layers: int   = 1        # 1 = single linear layer (fast)
    memory_hidden_dim: int   = -1       # -1 → auto = dim * 2

    # Per-token gate scales
    memory_lr:         float = 0.01     # multiplied onto theta gate output
    memory_momentum:   float = 0.9      # multiplied onto eta gate output

    # FFN inside LMMBlock
    ffn_dim:           int   = -1       # -1 → auto = dim * 4

    # Regularisation
    dropout:           float = 0.0

    # Nonlinearity for MemoryMLP
    activation:        str   = "silu"

    # Weight init
    init_std:          float = 0.02

    # Window size kept for interface compatibility (unused in paper-faithful version)
    window_size:       int   = 64

    def __post_init__(self):
        if self.memory_hidden_dim < 0:
            self.memory_hidden_dim = self.dim * 2
        if self.ffn_dim < 0:
            self.ffn_dim = self.dim * 4