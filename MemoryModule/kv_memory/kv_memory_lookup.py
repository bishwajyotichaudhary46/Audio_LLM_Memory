import torch
from torch import nn

class KVMemory(nn.Module):
    def __init__(self, memory_size, key_dim, value_dim):
        super(KVMemory, self).__init__()
        self.memory_size = memory_size  # 4000
        self.key_dim = key_dim          # 768
        self.value_dim = value_dim      # 768

        # Key Matrix: (768, 4000) — matches diagram
        self.keys = nn.Parameter(torch.zeros(key_dim, memory_size))
        # Values Matrix: (4000, 768) — matches diagram
        self.values = nn.Parameter(torch.zeros(memory_size, value_dim))

    def forward(self, query_key):
        # query_key shape: (1, 768) or (768,)
        # h · k → (1, 768) @ (768, 4000) = (1, 4000) 
        similarity = torch.matmul(query_key.unsqueeze(0), self.keys)  # (1, 4000)

        # Softmax over 4000 memory slots 
        similarity = torch.softmax(similarity, dim=-1)  # (1, 4000)

        # (1, 4000) @ (4000, 768) → (1, 768) = Δhidden state 
        value = torch.matmul(similarity, self.values).squeeze(0)  # (768,)

        return value