import torch
import torch.nn as nn
class LinearAttentionMem(nn.Module):
    def __init__(
        self,
        n_text_state: int,
        n_heads: int = 8,
        bias: bool = True,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_text_state = n_text_state

        # Projections
        self.keys_proj   = nn.Linear(n_text_state, self.n_text_state * n_heads, bias=bias)
        self.values_proj = nn.Linear(n_text_state, self.n_text_state * n_heads, bias=bias)

        # foget Projections
        self.forget = nn.Linear(n_text_state, self.n_heads )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ [B, S, n_text_state] → [B, S, H, D] """
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.n_text_state)


    def forward(self, x: torch.Tensor, M: torch.Tensor, router_weight: torch.Tensor, Mem_id):
        """
        Args:
            x     : [B, S, n_text_state]     current input chunk
            state : [B, H, D, D] or None  previous memory matrix (S_t = ∑ k v^T)

        Returns:
            output: [B, S, n_text_state]
            new_state: [B, H, D, D]
        """
        B, S, _ = x.shape

        # Project  K, V
        k = self.keys_proj(x)
        v = self.values_proj(x)

        #print("after projection",k.shape)

        # Split heads
        k = self._split_heads(k)                        # [B, S, H, D]
        v = self._split_heads(v)                        # [B, S, H, D]


        # Linear Attention Update
        for t in range(S):
            # Compute Forget
            forget = torch.sigmoid(self.forget(x))

            # Compute Memories Weight
            memory_weight = router_weight[:,t,Mem_id, None, None,None]

            # Update memory M
            M = forget[:,t,:, None, None]*M + memory_weight * torch.einsum('b h k, b h v -> b h k v', k[:, t], v[:, t])
            # Compute output at time t
            # o[:, t] = torch.einsum('b h k v, b h k -> b h v', M, q[:, t])

        return M