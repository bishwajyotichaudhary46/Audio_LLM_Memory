import torch
import torch.nn as nn
class MOM(nn.Module):
    def __init__(
        self,
        n_text_state,
        n_heads,
        num_memories,
        bias_term_adjust,
        LinearAttentionMem,
        topk=2

    ):
        super().__init__()

        self.n_text_state = n_text_state
        self.n_heads = n_heads
        self.num_memories = num_memories
        self.topk = topk
        self.bias_term_adjust = bias_term_adjust

        # Projections
        self.query_proj = nn.Linear(n_text_state, n_text_state * n_heads)

        # Output projection
        self.W = nn.Linear(n_text_state * n_heads, n_text_state)

        # Router
        self.router = nn.Linear(n_text_state, num_memories)

        # Bias Initialization
        self.register_buffer("bias", torch.zeros(32, num_memories))

        # Create a LinearAttention module per memory slot
        self.mult_mem = nn.ModuleList(
            [LinearAttentionMem(n_text_state=self.n_text_state,
                                n_heads= self.n_heads) for _ in range(num_memories)]
        )
        self.softmax = nn.Softmax(dim=-1)

    def split_head(self, t):
        """
        [B, S, H*D] -> [B, S, H, D]
        """
        B, S, _ = t.shape
        return t.view(B, S, self.n_heads, self.n_text_state)

    def mixed_head(self, t):
        """
        [B, S, H, D]->[B, S, H*D]
        """
        B, S, _,_ = t.shape
        return t.view(B, S, self.n_heads*self.n_text_state)
    


    def forward(self, x):
        """
        x: [B, S, n_text_state]
        M: [B, H*NumMem, K, V]
        """
        B, S,_ = x.shape
        total_token_routed = self.topk * S
        avg_load_per_expert = total_token_routed /self.num_memories
        # Compute projections
        q = self.query_proj(x)
        q = self.split_head(q)                # [B, S, H, D]
        _,_,H,D = q.shape

        # Router
        router_weight = self.router(x)                          # [B, S, num_memories]
        routing_weight, self.bias[:B] = self.bias_term_adjust(router_weight,
                                                     self.softmax,
                                                     avg_load_per_expert, self.num_memories, self.bias[:B])

        # M emory Temp
        memory_outputs = []

        # Run selected memory modules only
        for mem_id in range(self.num_memories):
            # Extract prev mem for this memory: [B,H,K,V]
            # M_t = M[:B,mem_id*self.n_heads:(mem_id+1)*self.n_heads, :,:]
            # Call attention module
            M_t = self.mult_mem[mem_id](x,routing_weight, mem_id)

            memory_outputs.append(M_t)


        # Sum all memory contributions
        final_memory = torch.stack(memory_outputs, dim=0).sum(0)   # [B,H,K,V]

        o = torch.zeros((B, S, H, D),  dtype=q.dtype).to(q.device)

        for t in range(S):
            o[:, t] = torch.einsum('b h k v, b h d -> b h v', final_memory, q[:, t])


        # Final output projection
        final_output = self.W(self.mixed_head(o))

        # # Merge heads and project
        # B, S, H, D = final_output.shape
        # final_output = final_output.view(B, S, H * D)
        # final_output = self.atten_out_proj(final_output)

        return final_output
