
from torch import nn
import random

class AdaMix(nn.Module):
    def __init__(self, hidden_size, bottleneck_dim, M=4):
        super().__init__()
        self.M = M  # number of adapters

        # Create M adapters per layer
        self.W_down = nn.ModuleList([
            nn.Linear(hidden_size, bottleneck_dim) for _ in range(M)
        ])
        self.W_up = nn.ModuleList([
            nn.Linear(bottleneck_dim, hidden_size) for _ in range(M)
        ])

    def forward(self, x):
        # Randomly pick one W_down and one W_up
        j = random.randint(0, self.M - 1)
        k = random.randint(0, self.M - 1)

        # Apply adapter transformation
        out = x + self.W_up[j](F.relu(self.W_down[k](x)))
        return out