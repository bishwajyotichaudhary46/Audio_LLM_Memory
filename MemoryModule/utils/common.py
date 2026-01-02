import torch
@torch.no_grad()
def bias_term_adjust(router_weight, softmax, avg_load_per_expert, num_memories, bias, u=0.001):
    """
    router_weight: [B, S, E]   (E = num_memories)
    bias:          [1, E] or [B, E]
    """

    # Get topk values & indices ----
    router_load_weight, top_memories_indx = torch.topk(
        router_weight, 2, dim=-1
    )


    # Scatter into full expert dimension ----
    routing_weights_full = torch.full(
        (router_weight.shape[0], router_weight.shape[1], num_memories),
        float('-inf'),
        device=router_weight.device,
        dtype=router_weight.dtype
    ).scatter(-1, top_memories_indx, router_load_weight)

    # Normalize
    routing_weights_full_norm = softmax(routing_weights_full)

    # Count routed tokens per expert
    token_routed_each_expert = torch.count_nonzero(routing_weights_full_norm, dim=-2)

    # Compute load violation
    load_violation = avg_load_per_expert - token_routed_each_expert
    # Shape: [B, E]

    # Aggregate over batch
    # Instead of updating using last batch element: load_violation[-1]
    # we do MEAN over batch = batch gradient estimate
    load_signal = load_violation.mean(dim=0)     # shape [E]

    # compute sign
    direction = torch.sign(load_signal)

    # Apply router weight update (vectorized)
    # router_weight[:,:,idx] += load * bias[:,idx]
    router_weight = router_weight + direction.unsqueeze(0).unsqueeze(0) * bias

    # Bias update like gradient descent
    # bias = bias + u * load_signal
    bias = bias + u * direction.unsqueeze(0)

    # Recompute new routing weights
    router_load_weight, top_memories_indx = torch.topk(
        router_weight, 2, dim=-1
    )

    routing_weights_full = torch.full(
        (router_weight.shape[0], router_weight.shape[1], num_memories),
        float('-inf'),
        device=router_weight.device,
        dtype=router_weight.dtype
    ).scatter(-1, top_memories_indx, router_load_weight)

    routing_weights_full_norm = softmax(routing_weights_full)

    return routing_weights_full_norm, bias