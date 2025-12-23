import torch
import torch.nn as nn

class MHARouting(nn.Module):
    def __init__(
        self,
        n_text_state: int,
        n_heads: int = 8,
        prenorm: bool = True,
    ):
        super().__init__()

        self.n_text_state = n_text_state
        self.n_heads = n_heads

        # Multihead Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=n_text_state,
            num_heads=n_heads,
            batch_first=True
        )

        # LayerNorm
        self.input_norm = nn.LayerNorm(n_text_state) if prenorm else nn.Identity()
        self.output_norm = nn.Identity() if prenorm else nn.LayerNorm(n_text_state)

        # Router
        self.router_norm = nn.LayerNorm(n_text_state)
        self.router_proj = nn.Linear(n_text_state, 1)

        with torch.no_grad():
            self.router_proj.weight.zero_()
            self.router_proj.bias.fill_(-10.0)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # Pre-LN
        x_norm = self.input_norm(x)

        # MHA
        y, _ = self.mha(x_norm, x_norm, x_norm, need_weights=False)

        # Residual + LN
        y = x + y
        y = self.output_norm(y)

        # Router (bounded input)
        y = self.router_norm(y)
        logits = self.router_proj(y)

        # Sigmoid gate
        gate = torch.sigmoid(logits)

        return gate
