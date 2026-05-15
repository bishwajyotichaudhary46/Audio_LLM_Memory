from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FastMemoryState:
    weights: List[torch.Tensor]
    momentum: List[torch.Tensor]

    def detach(self) -> "FastMemoryState":
        return FastMemoryState(
            weights=[w.detach() for w in self.weights],
            momentum=[m.detach() for m in self.momentum],
        )

    def clone(self) -> "FastMemoryState":
        return FastMemoryState(
            weights=[w.clone() for w in self.weights],
            momentum=[m.clone() for m in self.momentum],
        )


class AssociativeMemory(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_layers: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        layers = []

        if num_layers <= 1:
            layers.append(nn.Linear(dim, dim, bias=False))
        else:
            layers.append(nn.Linear(dim, hidden_dim, bias=False))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.Linear(hidden_dim, dim, bias=False))

        self.layers = nn.ModuleList(layers)
        self.act = nn.SiLU()

        for layer in self.layers:
            nn.init.normal_(layer.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h

    def cloned_weights(self) -> List[torch.Tensor]:
        return [layer.weight.detach().clone() for layer in self.layers]

    def load_weights(self, weights: List[torch.Tensor]) -> None:
        for layer, w in zip(self.layers, weights):
            layer.weight.data.copy_(w)

    def assoc_loss(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        pred = self(k)
        return F.mse_loss(pred, v)


class TitansStyleMemory(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_memory_layers: int = 2,
        memory_lr: float = 0.1,
        memory_momentum: float = 0.9,
        init_std: float = 0.02,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.memory_lr = memory_lr
        self.memory_momentum = memory_momentum
        self.eps = eps

        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.memory = AssociativeMemory(
            dim=dim,
            hidden_dim=hidden_dim,
            num_layers=num_memory_layers,
            init_std=init_std,
        )

        self.decay_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.lr_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.mom_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        for mod in [self.k_proj, self.v_proj, self.q_proj, self.out_proj]:
            nn.init.normal_(mod.weight, std=init_std)

        # internal dynamic memory state
        self._state: Optional[FastMemoryState] = None

    def init_state(self, device: torch.device) -> FastMemoryState:
        weights = self.memory.cloned_weights()
        weights = [w.to(device) for w in weights]
        momentum = [torch.zeros_like(w) for w in weights]
        return FastMemoryState(weights=weights, momentum=momentum)

    def reset_state(self, device: torch.device) -> None:
        self._state = self.init_state(device)

    def clear_state(self) -> None:
        self._state = None

    def has_state(self) -> bool:
        return self._state is not None

    def get_state(self) -> Optional[FastMemoryState]:
        return None if self._state is None else self._state.clone()

    def set_state(self, state: Optional[FastMemoryState]) -> None:
        self._state = None if state is None else state.detach()

    def _resolve_state(
        self,
        x: torch.Tensor,
        state: Optional[FastMemoryState] = None,
    ) -> FastMemoryState:
        if state is not None:
            return state
        if self._state is not None:
            return self._state
        return self.init_state(x.device)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(F.silu(x), p=2, dim=-1, eps=self.eps)

    def _compute_grads(self, k: torch.Tensor, v: torch.Tensor) -> List[torch.Tensor]:
        with torch.enable_grad():
            params = list(self.memory.parameters())

            for p in params:
                p.requires_grad_(True)

            k_in = k.detach().requires_grad_(True)
            v_tgt = v.detach()

            loss = self.memory.assoc_loss(k_in, v_tgt)

            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=False,
                allow_unused=True,
            )

            for p in params:
                p.requires_grad_(False)

        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.memory.parameters())
        ]

    def retrieve(
        self,
        queries: torch.Tensor,
        state: Optional[FastMemoryState] = None,
    ) -> torch.Tensor:
        """
        Retrieve from memory without updating internal state.
        """
        resolved_state = self._resolve_state(queries, state)
        self.memory.load_weights(resolved_state.weights)

        q = self._normalize(self.q_proj(queries))
        retrieved = self.memory(q)
        return self.out_proj(retrieved)

    def forward(
        self,
        x: torch.Tensor,
        kv_source: torch.Tensor,
        state: Optional[FastMemoryState] = None,
        update_memory: bool = True,
        return_state: bool = True,
        store_internal_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[FastMemoryState]]:
        """
        Args:
            x: query-side tensor (batch, seq, dim)
            kv_source: source tensor used to build key/value
            state: optional external state. If None, internal state is used.
            update_memory: whether to update fast memory
            return_state: whether to return state
            store_internal_state: whether to overwrite self._state after update

        Returns:
            output, state (optional)
        """
        resolved_state = self._resolve_state(x, state)

        # load current memory weights
        self.memory.load_weights(resolved_state.weights)

        q = self._normalize(self.q_proj(x))
        k = self._normalize(self.k_proj(kv_source))
        v = F.silu(self.v_proj(kv_source))

        # retrieve before update
        retrieved = self.memory(q)
        output = self.out_proj(retrieved)

        if not update_memory:
            out_state = resolved_state.detach()
            if store_internal_state:
                self._state = out_state
            if return_state:
                return output, out_state
            return output, None

        pooled = x.mean(dim=1, keepdim=True)

        alpha = self.decay_gate(pooled).mean()
        theta = self.lr_gate(pooled).mean() * self.memory_lr
        eta = self.mom_gate(pooled).mean() * self.memory_momentum

        grads = self._compute_grads(k, v)

        new_momentum: List[torch.Tensor] = []
        new_weights: List[torch.Tensor] = []

        for w, m, g in zip(resolved_state.weights, resolved_state.momentum, grads):
            surprise = eta * m - theta * g
            w_new = (1.0 - alpha) * w + surprise
            new_momentum.append(surprise)
            new_weights.append(w_new)

        new_state = FastMemoryState(
            weights=new_weights,
            momentum=new_momentum,
        ).detach()

        if store_internal_state:
            self._state = new_state

        if return_state:
            return output, new_state
        return output, None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate(x)) * self.up(x)
        h = self.dropout(h)
        return self.down(h)


class TitansMemoryBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mem_hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.memory = TitansStyleMemory(
            dim=dim,
            hidden_dim=mem_hidden_dim,
            num_memory_layers=2,
            memory_lr=0.1,
            memory_momentum=0.9,
        )

        self.ffn = FeedForward(dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def reset_state(self, device: torch.device) -> None:
        self.memory.reset_state(device)

    def clear_state(self) -> None:
        self.memory.clear_state()

    def get_state(self) -> Optional[FastMemoryState]:
        return self.memory.get_state()

    def set_state(self, state: Optional[FastMemoryState]) -> None:
        self.memory.set_state(state)

    def retrieve(
        self,
        queries: torch.Tensor,
        state: Optional[FastMemoryState] = None,
    ) -> torch.Tensor:
        return self.memory.retrieve(self.norm1(queries), state=state)

    def forward(
        self,
        x: torch.Tensor,
        kv_source: torch.Tensor,
        update_memory: bool = True,
        return_state: bool = False,
        store_internal_state: bool = True,
    ):
        mem_out, new_state = self.memory(
            self.norm1(x),
            kv_source=kv_source,
            state=None,  # use internal state by default
            update_memory=update_memory,
            return_state=True,
            store_internal_state=store_internal_state,
        )

        x = x + self.dropout(mem_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        if return_state:
            return x, new_state
        return x