import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.functional import normalize

from MemoryModule.conponents.context_neural_memory import ContextNeuralMemory

class ContextMemoryLayer(nn.Module):
    def __init__(self, hidden_dim, seq_len, pm_len, n_layers_nmm = 2):
        super().__init__()
        self.seq_len = seq_len
        self.pm_len = pm_len
        self.hidden_dim = hidden_dim
        self.inter_dim = (pm_len + 2 * hidden_dim)

        # Persistent memory weights
        self.persistent_memory = nn.Parameter(torch.randn((pm_len, self.hidden_dim)))

        # The attention-based processing core
        self.att_layer = nn.MultiheadAttention(
            embed_dim = hidden_dim, 
            num_heads = 8, 
            dropout=0.2, 
            bias=True, 
            add_bias_kv=False, 
            add_zero_attn=False,    
        )
        
        # Mapping to queries for retrieving from NMM
        self.Q = nn.Linear(hidden_dim, hidden_dim)

        # The Neural Memory Module (NMM)
        self.nm_module = ContextNeuralMemory(
            emb_dim = hidden_dim,
            n_layers = n_layers_nmm,
            hidden_dim = 2 * hidden_dim,
        )

        # self.final_layer = nn.Linear(hidden_dim, seq_len * hidden_dim)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        # self.deivce = None

        # self.outer_params = [self.persistent_memory] + list(self.Q.parameters()) + list(self.final_layer.parameters()) + list(self.att_layer.parameters())
        self.outer_params = list(self.Q.parameters())

    def forward(self, x, alpha, theta, eta):
        # x.shape = (batch_size, seq_len, hidden_dim)
        batch_size = x.shape[0]

        # Retrieve knowledge from the NMM
        queries = self.silu(normalize(self.Q(x.view(-1, self.hidden_dim))))
        nmm_vals = self.nm_module.retrieve(queries).view(batch_size, -1, self.hidden_dim)
        # print("nnm_val", nmm_vals.shape)

        # # # Concatenate persistent and long-term memory to the beginning
        # pm_expanded = self.persistent_memory.unsqueeze(0).expand(x.shape[0], -1, -1)
        # print("shape of pm_expanddedz:", pm_expanded.shape)
        # x = torch.cat([nmm_vals, x], dim=1)
        x = nmm_vals + x

        

        # print(x.shape)

        # Pass through the attention layer
        # print("shaper of x ",x.shape)
        # print("shaper of attention ",self.inter_dim )
        # x = self.att_layer(query=x, key=x, value=x)
        # x = self.silu(self.att_layer(query=x, key=x, value=x)[0])
        # print("shape of x after attention:", x)
                      
                    #   x.view(-1, self.inter_dim * self.hidden_dim))
        # x = self.final_layer(x).view(-1, self.hidden_dim)

        # Update the NMM
        _, new_params = self.nm_module.update(x, alpha, theta, eta)

        # print("new params", new_params,)

        # Retrieve new information
        y = functional_call(self.nm_module, new_params, normalize(self.Q(x)))


        # print("x shape", x.shape)

        # Gate the output using the retrieved memory
        return (x * self.sigmoid(y))