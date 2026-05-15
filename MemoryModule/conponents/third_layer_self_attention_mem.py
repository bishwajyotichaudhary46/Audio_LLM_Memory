"""
Standalone Whisper Self-Attention (no KV cache, no cross-attention)
=====================================================================
Stripped down from HuggingFace's WhisperAttention to the pure
self-attention path: key_value_states=None, past_key_value=None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple



# Helper
def _shape(tensor: torch.Tensor, seq_len: int, bsz: int,
           num_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape (bsz, seq, embed) → (bsz, heads, seq, head_dim)."""
    return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()



# Self-Attention Module
class SelfAttention(nn.Module):
    """
    Multi-head self-attention (no KV cache, no cross-attention).

    All three of Q, K, V are projected from the SAME hidden_states tensor,
    so every position attends to every other position in the sequence.

    Args:
        embed_dim  : Total embedding dimension (must be divisible by num_heads).
        num_heads  : Number of attention heads.
        dropout    : Dropout probability on attention weights. Default: 0.0.
        bias       : Whether to add bias to V/Q/out projections. Default: True.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout
        self.scaling   = self.head_dim ** -0.5   # 1 / sqrt(head_dim)

        # Projections 
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states    : (batch, seq_len, embed_dim)
            attention_mask   : (batch, 1, tgt_len, src_len)  additive mask
                               (0 = keep, large negative = ignore).
            layer_head_mask  : (num_heads,) per-head scaling factor.
            output_attentions: Return attention weights alongside output.

        Returns:
            attn_output  : (batch, seq_len, embed_dim)
            attn_weights : (batch, num_heads, seq_len, seq_len)  or  None
        """
        bsz, tgt_len, _ = hidden_states.size()

        # Linear projections
        # Scale queries before matmul (numerically equivalent to post-scale)
        query_states = _shape(
            self.q_proj(hidden_states) * self.scaling,
            tgt_len, bsz, self.num_heads, self.head_dim,
        )  # (bsz, heads, tgt_len, head_dim)

        key_states = _shape(
            self.k_proj(hidden_states),
            -1, bsz, self.num_heads, self.head_dim,
        )  # (bsz, heads, tgt_len, head_dim)

        value_states = _shape(
            self.v_proj(hidden_states),
            -1, bsz, self.num_heads, self.head_dim,
        )  # (bsz, heads, tgt_len, head_dim)

        # Scaled dot-product attention scores 
        # (bsz, heads, tgt_len, tgt_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        # Additive attention mask (padding / causal) 
        if attention_mask is not None:
            # attention_mask may be larger; slice to key length
            causal_mask  = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax over key dimension 
        attn_weights = F.softmax(attn_weights, dim=-1)


        # Optional per-head mask 
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"layer_head_mask must be of shape ({self.num_heads},), "
                    f"got {layer_head_mask.size()}."
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        # Dropout on attention probabilities
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Weighted sum of values
        # (bsz, heads, tgt_len, head_dim)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, tgt_len, self.head_dim)}, "
                f"got {attn_output.size()}."
            )

        # Merge heads + output projection 
        attn_output = attn_output.transpose(1, 2)                      # (bsz, tgt_len, heads, head_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)  # (bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)                        # (bsz, tgt_len, embed_dim)

        attn_weights_out = attn_weights if output_attentions else None
        return attn_output, attn_weights_out



# Quick smoke-test

if __name__ == "__main__":
    torch.manual_seed(0)

    BATCH, SEQ, EMBED, HEADS = 2, 10, 512, 8

    model = SelfAttention(embed_dim=EMBED, num_heads=HEADS, dropout=0.1)
    model.eval()

    x = torch.randn(BATCH, SEQ, EMBED)

    # no mask 
    out, _ = model(x)
    assert out.shape == (BATCH, SEQ, EMBED), f"Unexpected shape: {out.shape}"
    print(f"[OK] output shape : {out.shape}")

    # with causal mask 
    causal = torch.zeros(BATCH, 1, SEQ, SEQ)
    mask   = torch.full((SEQ, SEQ), float("-inf")).triu(diagonal=1)
    causal += mask.unsqueeze(0).unsqueeze(0)
    out_masked, weights = model(x, attention_mask=causal, output_attentions=True)
    assert out_masked.shape == (BATCH, SEQ, EMBED)
    assert weights.shape    == (BATCH, HEADS, SEQ, SEQ)
    print(f"[OK] masked output: {out_masked.shape}")
    print(f"[OK] attn weights : {weights.shape}")
    print("\nAll assertions passed.")