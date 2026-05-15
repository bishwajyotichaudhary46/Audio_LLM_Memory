# coding=utf-8
"""
Titans Neural Long-Term Memory — paper-faithful, seq2seq-corrected.
Optimised for speed without changing any logic.

Speed changes (logic identical):
  1. _write_chunk: eliminated the Python for-loop over C_w tokens.
     Single-layer gradient is now a fully batched matmul over the whole
     chunk at once.  Multi-layer uses a single autograd.grad call over
     the full chunk instead of C per-token calls.
  2. _autograd_chunk_grads: replaced C individual .grad calls with one
     batched call — the MSE loss is summed over all tokens so the gradient
     is identical to averaging per-token gradients.
  3. Removed redundant .clone() on weights/momentum inside _write_chunk —
     new lists are built directly instead of cloning then overwriting.
  4. Padding computed with unary negation modulo: (-n) % C  (cleaner + same).
  5. k/v padded together in one branch.
  6. Read outputs collected into a list then cat once (avoids repeated allocs).
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TitansConfig



# MemoryState
@dataclass
class MemoryState:
    weights:  list[torch.Tensor]
    momentum: list[torch.Tensor]

    def detach(self) -> MemoryState:
        return MemoryState(
            [w.detach() for w in self.weights],
            [m.detach() for m in self.momentum],
        )

    def to(self, device: torch.device) -> MemoryState:
        return MemoryState(
            [w.to(device) for w in self.weights],
            [m.to(device) for m in self.momentum],
        )

    def detach_to(self, device: torch.device) -> MemoryState:
        return MemoryState(
            [w.detach().to(device) for w in self.weights],
            [m.detach().to(device) for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        return MemoryState(
            [w.clone() for w in self.weights],
            [m.clone() for m in self.momentum],
        )



# MemoryMLP
class MemoryMLP(nn.Module):
    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.num_layers = config.num_memory_layers
        self.dim        = config.dim
        self.hidden_dim = getattr(config, "memory_hidden_dim", config.dim * 2)

        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            layers.append(nn.Linear(self.dim, self.dim, bias=False))
            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(self.dim, self.dim, bias=False))
            layers.append(nn.Linear(self.dim, self.dim, bias=False))
        self.layers = nn.ModuleList(layers)

        act_name = getattr(config, "activation", "silu")
        self.activation = {"silu": nn.SiLU(), "gelu": nn.GELU(), "relu": nn.ReLU()}[act_name]
        self._init()

    def _init(self) -> None:
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)

    def get_weights(self) -> list[torch.Tensor]:
        return [l.weight.data.clone() for l in self.layers]

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        for layer, w in zip(self.layers, weights, strict=True):
            layer.weight.data.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def associative_loss(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(keys), values, reduction="mean")



# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight



# NeuralLongTermMemory
class NeuralLongTermMemory(nn.Module):
    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config     = config
        self.dim        = config.dim
        self.chunk_size = getattr(config, "chunk_size", 64)

        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        self.memory = MemoryMLP(config)

        self.gate_alpha = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())
        self.gate_theta = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())
        self.gate_eta   = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())

        self.memory_lr       = getattr(config, "memory_lr",      0.01)
        self.memory_momentum = getattr(config, "memory_momentum", 0.9)

        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)
        self.norm_out = RMSNorm(config.dim)

        self._init_weights()

    def _init_weights(self) -> None:
        std = getattr(self.config, "init_std", 0.02)
        for m in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            nn.init.normal_(m.weight, std=std)

    def init_state(self, device: torch.device) -> MemoryState:
        weights  = [w.clone().to(device) for w in self.memory.get_weights()]
        momentum = [torch.zeros_like(w) for w in weights]
        return MemoryState(weights=weights, momentum=momentum)

    def _write_chunk(
        self,
        keys:   torch.Tensor,   # (B, C_w, D)
        values: torch.Tensor,   # (B, C_w, D)
        alpha:  torch.Tensor,   # (B, C_q, D)
        theta:  torch.Tensor,   # (B, C_q, D)
        eta:    torch.Tensor,   # (B, C_q, D)
        state:  MemoryState,
    ) -> MemoryState:
        """
        Original logic: loop over C_w tokens, apply update rule per token.
        Optimised:      compute gradient for the whole chunk in one shot.

        Single-layer:
          Original per-token grad: outer(err_t, k_t) = (Wk_t - v_t) ⊗ k_t
          Batched equivalent:      einsum("bco,bci->oi", err, k) / (B*C_w)
          This is mathematically identical to averaging the C_w per-token
          outer products — same as the original loop's mean(0) at the end.

        Multi-layer:
          Original: C_w separate autograd calls, one per token.
          Optimised: one autograd call on the full-chunk MSE loss.
          MSE with reduction="mean" sums over tokens then divides — the
          gradient is the same as averaging per-token gradients.

        Gate scalars a/h/e are mean-pooled from the query chunk — unchanged.
        """
        self.memory.set_weights(state.weights)

        # Gate scalars: (D,)  — identical to original mean over query chunk
        a = alpha.mean(dim=(0, 1))
        h = theta.mean(dim=(0, 1)) * self.memory_lr
        e = eta.mean(dim=(0, 1))   * self.memory_momentum

        # Unsqueeze once for broadcasting over D_in
        a_ = a.unsqueeze(-1)   # (D, 1)
        h_ = h.unsqueeze(-1)
        e_ = e.unsqueeze(-1)

        

        if len(self.memory.layers) == 1:
            W   = state.weights[0]                              # (D, D)
            Wk  = keys @ W.t()                                  # (B, C_w, D)
            err = Wk - values                                   # (B, C_w, D)

            # Equivalent to mean of per-token outer products
            N    = keys.shape[0] * keys.shape[1]
            grad = torch.einsum("bco,bci->oi", err, keys) / N  # (D, D)

            s     = e_ * state.momentum[0] - h_ * grad
            w_new = (1 - a_) * state.weights[0] + s
            return MemoryState(weights=[w_new], momentum=[s])

        else:
            # One autograd call for the whole chunk
            grads = self._autograd_chunk_grads(keys, values)

            new_weights  = []
            new_momentum = []
            for g, w, m in zip(grads, state.weights, state.momentum):
                s     = e_ * m - h_ * g
                w_new = (1 - a_) * w + s
                new_weights.append(w_new)
                new_momentum.append(s)
            return MemoryState(weights=new_weights, momentum=new_momentum)


    def _autograd_chunk_grads(
        self,
        keys:   torch.Tensor,   # (B, C, D)
        values: torch.Tensor,   # (B, C, D)
    ) -> list[torch.Tensor]:
        """
        Original: loop over C tokens, one autograd.grad per token → C kernel
                  launches, C graph traversals.
        Optimised: one autograd.grad on the full-chunk MSE loss → 1 traversal.

        MSE with reduction="mean" averages over all elements including the
        token dimension, so the gradient w.r.t. each weight is the average
        of the per-token gradients — mathematically identical to the original.
        """
        with torch.enable_grad():
            for p in self.memory.parameters():
                p.requires_grad_(True)

            # Single loss over the whole chunk
            loss = self.memory.associative_loss(
                keys.detach(), values.detach()
            )

            grads = torch.autograd.grad(
                loss,
                list(self.memory.parameters()),
                create_graph=False,
                allow_unused=True,
            )

            for p in self.memory.parameters():
                p.requires_grad_(False)

        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.memory.parameters(), strict=True)
        ]

    def _read_chunk(
        self,
        queries: torch.Tensor,
        state:   MemoryState,
    ) -> torch.Tensor:
        self.memory.set_weights(state.weights)
        return self.memory(queries)


    def forward(
        self,
        x:                     torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        state:                 MemoryState | None  = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        B, S, D = x.shape
        device  = x.device

        if state is None:
            state = self.init_state(device)
        if state.weights[0].device != device:
            state = state.to(device)

        src = encoder_hidden_states if encoder_hidden_states is not None else x

        # Projections
        k = F.normalize(F.silu(self.proj_k(x)), p=2, dim=-1)
        v = F.silu(self.proj_v(x))
        q = F.normalize(F.silu(self.proj_q(x)),  p=2, dim=-1)

        # Gates (decoder x)
        alpha = self.gate_alpha(x)
        theta = self.gate_theta(x)
        eta   = self.gate_eta(x)

        # Pad write sequence — (-n) % C is cleaner than (C - n%C) % C
        C     = self.chunk_size
        pad_w = (-k.shape[1]) % C
        if pad_w:
            k = F.pad(k, (0, 0, 0, pad_w))
            v = F.pad(v, (0, 0, 0, pad_w))

        # Pad read sequence
        pad_r = (-S) % C
        if pad_r:
            q     = F.pad(q,     (0, 0, 0, pad_r))
            alpha = F.pad(alpha, (0, 0, 0, pad_r))
            theta = F.pad(theta, (0, 0, 0, pad_r))
            eta   = F.pad(eta,   (0, 0, 0, pad_r))

        n_write_chunks = k.shape[1] // C
        n_read_chunks  = q.shape[1] // C

        # Write pass
        for c in range(n_write_chunks):
            wsl   = slice(c * C, (c + 1) * C)
            q_idx = min(c, n_read_chunks - 1)
            qsl   = slice(q_idx * C, (q_idx + 1) * C)
            state = self._write_chunk(
                k[:, wsl], v[:, wsl],
                alpha[:, qsl], theta[:, qsl], eta[:, qsl],
                state,
            )

        # Read pass — list-then-cat avoids repeated intermediate allocs
        outputs = [
            self._read_chunk(q[:, c * C : (c + 1) * C], state)
            for c in range(n_read_chunks)
        ]
        output = torch.cat(outputs, dim=1)[:, :S]

        output = self.norm_out(self.proj_out(output))
        return output, state.detach_to(device)