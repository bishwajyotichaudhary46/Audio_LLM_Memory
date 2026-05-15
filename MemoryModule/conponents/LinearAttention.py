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
        self.keys_proj   = nn.Linear(n_text_state, self.n_text_state, bias=bias)
        self.values_proj = nn.Linear(n_text_state, self.n_text_state, bias=bias)

        # foget Projections
        #self.forget = nn.Linear(n_text_state, self.n_heads)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ [B, S, n_text_state] → [B, S, H, D] """
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.n_text_state//self.n_heads)


    def forward(self, x: torch.Tensor, router_weight: torch.Tensor = None, Mem_id=None):
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
       

        with torch.no_grad():
            
            #print(router_weight[:,:,Mem_id].shape)
            # print(v.shape)
            kv = torch.einsum('bshk,bshv->bshkv', k, v)
            # print("kv",kv.shape)
            # print("forget", forget.shape)
            #contribute = router_weight[:,:,Mem_id, None, None, None]* kv
            if router_weight is None:
                M =  kv.float().mean(dim=1)
            else:
                M = torch.einsum('bs,bshkv->bshkv', router_weight[:,:,Mem_id], kv).float().mean(dim=1)

        return M