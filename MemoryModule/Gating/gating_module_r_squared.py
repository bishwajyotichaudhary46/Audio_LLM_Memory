import torch.nn as nn
import torch.nn.functional as F
import torch


class Gating(nn.Module):
    """
    Token-level gating head on top of decoder hidden states.

    Projects each token's hidden state through a two-layer MLP
    and returns a per-token gate value in [0, 1].

    Args:
        config: must expose `config.d_model` (int)
    """

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=True)
        self.fc2 = nn.Linear(4 * d_model, 1,              bias=True)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, T, d_model)  — last decoder layer output

        Returns:
            gate_output:   (B, T, 1)        — values in [0, 1]
        """
        h           = F.relu(self.fc1(hidden_states))   # (B, T, 4*d_model)
        gate_logits = self.fc2(h * h)                  # (B, T, 1)  — squared activation
        gate_output = torch.sigmoid(gate_logits)         # (B, T, 1)  ∈ [0, 1]
        return gate_output