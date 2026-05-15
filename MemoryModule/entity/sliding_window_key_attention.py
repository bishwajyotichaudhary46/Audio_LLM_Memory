"""
Sliding Window Key-Queue Attention Mechanism
=============================================

Architecture:
    1. Project hidden states (n, embed_dim) -> keys (n, 128)
    2. Maintain a causal FIFO queue of size 4 per token (sliding window)
    3. Flatten each token's queue context: (4, 128) -> (512,)
    4. Scaled dot product with learnable W1 (512, 2048) + softmax
    5. Project back with W2 (2048, embed_dim) -> (n, embed_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlidingWindowKeyQueueAttention(nn.Module):
    """
    Sliding Window Key-Queue Attention.

    For each token at position i, retrieves the last `queue_size` keys
    (including the current token's key), pads with a learnable pad embedding
    at the start when context is insufficient, flattens the window, and
    projects through a 2-layer attention-style transformation.

    Args:
        embed_dim   : Hidden state dimension (e.g. 768)
        key_dim     : Projected key dimension (default: 128)
        queue_size  : Sliding window / queue length (default: 4)
        inner_dim   : Intermediate projection dimension (default: 2048)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        key_dim: int = 128,
        queue_size: int = 4,
        inner_dim: int = 2048,
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.key_dim    = key_dim
        self.queue_size = queue_size
        self.inner_dim  = inner_dim
        self.flat_dim   = queue_size * key_dim          # 4 * 128 = 512

        # Layers 
        # Project hidden state -> key
        self.key_proj = nn.Linear(embed_dim, key_dim, bias=False)

        # Learnable pad token embedding in key space
        #  Shape: (1, key_dim) — represents "no token yet"
        self.pad_key = nn.Parameter(torch.zeros(1, key_dim))
        nn.init.normal_(self.pad_key, mean=0.0, std=0.02)

        # queries: (flat_dim -> inner_dim)  — scale + softmax applied after
        self.W1 = nn.Linear(self.flat_dim, inner_dim, bias=False)

        # 4. W2: (inner_dim -> embed_dim)
        self.W2 = nn.Linear(inner_dim, embed_dim, bias=False)

        # Scale factor: 1 / sqrt(flat_dim)
        self.scale = 1.0 / math.sqrt(self.flat_dim)


    # Queue builder  (pure-tensor, no Python loops over batch)
    def _build_queue_contexts(self, keys: torch.Tensor) -> torch.Tensor:
        """
        For every position i in the sequence, collect the window:
            [key_{i - queue_size + 1}, ..., key_{i - 1}, key_i]
        Positions before the sequence start are filled with self.pad_key.

        Args:
            keys : (batch, n, key_dim)

        Returns:
            windows : (batch, n, queue_size, key_dim)
        """
        B, n, D = keys.shape
        Q = self.queue_size

        # Prepend (Q-1) pad keys so that position 0 sees Q-1 pads + itself
        # pad_key: (1, 1, D) -> expand to (B, Q-1, D)
        pad = self.pad_key.unsqueeze(0).expand(B, Q - 1, D)   # (B, Q-1, D)
        padded = torch.cat([pad, keys], dim=1)                 # (B, Q-1+n, D)

        # Use unfold to extract sliding windows of length Q
        # unfold(dimension, size, step)
        # padded: (B, Q-1+n, D) -> unfold on dim=1
        windows = padded.unfold(1, Q, 1)                       # (B, n, D, Q)
        windows = windows.permute(0, 1, 3, 2)                  # (B, n, Q, D)

        return windows                                          # (B, n, Q, D)

    # Forward
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden : (batch, n, embed_dim)  or  (n, embed_dim) for unbatched

        Returns:
            out    : same shape as hidden — (batch, n, embed_dim)
        """
        unbatched = hidden.dim() == 2
        if unbatched:
            hidden = hidden.unsqueeze(0)            # (1, n, embed_dim)

        B, n, _ = hidden.shape

        # project to keys
        keys = self.key_proj(hidden)                # (B, n, key_dim)

        # build causal sliding-window contexts
        windows = self._build_queue_contexts(keys)  # (B, n, Q, key_dim)

        # flatten window
        flat = windows.reshape(B, n, self.flat_dim) # (B, n, flat_dim)

        # W1 projection, scale, softmax
        scores = self.W1(flat)                      # (B, n, inner_dim)
        scores = scores * self.scale                # scaled
        attn   = F.softmax(scores, dim=-1)          # (B, n, inner_dim)

        # W2 projection back to embed_dim 
        out = self.W2(attn)                         # (B, n, embed_dim)

        if unbatched:
            out = out.squeeze(0)                    # (n, embed_dim)

        return out



# Residual wrapper  (optional — plug into a transformer block easily)
class SlidingWindowKeyQueueLayer(nn.Module):
    """
    Wraps SlidingWindowKeyQueueAttention with:
        - LayerNorm (pre-norm style)
        - Residual connection

    Args:
        embed_dim, key_dim, queue_size, inner_dim : passed to core module
    """

    def __init__(self, embed_dim=768, key_dim=128, queue_size=4, inner_dim=2048):
        super().__init__()
        self.norm   = nn.LayerNorm(embed_dim)
        self.attn   = SlidingWindowKeyQueueAttention(
            embed_dim=embed_dim,
            key_dim=key_dim,
            queue_size=queue_size,
            inner_dim=inner_dim,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.attn(self.norm(hidden))



# Quick sanity check
if __name__ == "__main__":
    torch.manual_seed(42)

    embed_dim  = 768
    key_dim    = 128
    queue_size = 4
    inner_dim  = 2048
    batch_size = 2
    seq_len    = 10

    # --- Core module ---
    model = SlidingWindowKeyQueueAttention(
        embed_dim=embed_dim,
        key_dim=key_dim,
        queue_size=queue_size,
        inner_dim=inner_dim,
    )

    hidden = torch.randn(batch_size, seq_len, embed_dim)
    out    = model(hidden)

    print("=" * 55)
    print("  Sliding Window Key-Queue Attention — Shape Check")
    print("=" * 55)
    print(f"  Input  hidden : {tuple(hidden.shape)}")
    print(f"  Output hidden : {tuple(out.shape)}")
    print()

    # Verify per-token queue windows for batch=0
    keys    = model.key_proj(hidden)                     # (B, n, 128)
    windows = model._build_queue_contexts(keys)          # (B, n, 4, 128)
    flat    = windows.reshape(batch_size, seq_len, -1)   # (B, n, 512)

    print("  Per-token queue (batch 0, first 5 tokens):")
    print(f"  {'Token':<8} {'Queue window (positions shown as token idx, -1=pad)'}")
    pad_key_val = model.pad_key.detach()
    for i in range(min(5, seq_len)):
        window_keys = windows[0, i]                      # (4, 128)
        labels = []
        for q in range(queue_size):
            token_idx = i - (queue_size - 1 - q)
            labels.append("pad" if token_idx < 0 else f"t{token_idx}")
        print(f"  token {i:<3}  ->  queue: {labels}")

    print()
    print(f"  flat context shape : {tuple(flat.shape)}")

    # Residual layer
    layer = SlidingWindowKeyQueueLayer(
        embed_dim=embed_dim,
        key_dim=key_dim,
        queue_size=queue_size,
        inner_dim=inner_dim,
    )
    out2 = layer(hidden)
    print(f"\n  Residual layer output : {tuple(out2.shape)}")
    print("=" * 55)

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable parameters : {total:,}")
    print("=" * 55)