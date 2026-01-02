import torch
def create_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values=None,
    position_ids=None,
):
    """
    Performance-safe causal mask.
    Returns:
        - None → use is_causal=True (Flash / SDPA fast path)
        - 4D padding mask only when needed
    """

    batch_size, q_len, _ = input_embeds.shape

    # ----------------------------------------
    # 1. Fastest path: no padding at all
    # ----------------------------------------
    if attention_mask is None:
        return None

    # ----------------------------------------
    # 2. If attention_mask is already 4D → trust it
    # ----------------------------------------
    if attention_mask.dim() == 4:
        return attention_mask

    # ----------------------------------------
    # 3. Check if padding exists
    # ----------------------------------------
    # attention_mask: [B, K]
    if attention_mask.all():
        return None  # still safe to skip

    # ----------------------------------------
    # 4. KV-cache aware padding mask
    # ----------------------------------------
    kv_len = attention_mask.shape[-1]

    # Shape: [B, 1, 1, K]
    causal_mask = attention_mask[:, None, None, :kv_len]

    return causal_mask
