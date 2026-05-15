# coding=utf-8
"""
Titans Neural Long-Term Memory — chunk-free version.
Write and read operate over the full sequence in one shot.
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
        self.config = config
        self.dim    = config.dim

        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        self.memory = MemoryMLP(config)

        self.gate_alpha = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())
        self.gate_theta = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())
        self.gate_eta   = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())

        self.memory_lr       = getattr(config, "memory_lr",       0.01)
        self.memory_momentum = getattr(config, "memory_momentum",  0.9)

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

    # ------------------------------------------------------------------
    # Write: full sequence at once (no chunking)
    # ------------------------------------------------------------------
    def _write(
        self,
        keys:   torch.Tensor,   # (B, S, D)
        values: torch.Tensor,   # (B, S, D)
        alpha:  torch.Tensor,   # (B, S, D)
        theta:  torch.Tensor,   # (B, S, D)
        eta:    torch.Tensor,   # (B, S, D)
        state:  MemoryState,
    ) -> MemoryState:
        self.memory.set_weights(state.weights)

        # Gate scalars: mean over batch and full sequence → (D,)
        a = alpha.mean(dim=(0, 1))
        h = theta.mean(dim=(0, 1)) * self.memory_lr
        e = eta.mean(dim=(0, 1))   * self.memory_momentum

        a_ = a.unsqueeze(-1)  # (D, 1) — broadcast over D_in
        h_ = h.unsqueeze(-1)
        e_ = e.unsqueeze(-1)

        if len(self.memory.layers) == 1:
            W   = state.weights[0]                               # (D, D)
            Wk  = keys @ W.t()                                   # (B, S, D)
            err = Wk - values                                    # (B, S, D)

            N    = keys.shape[0] * keys.shape[1]                 # B * S
            grad = torch.einsum("bso,bsi->oi", err, keys) / N   # (D, D)

            s     = e_ * state.momentum[0] - h_ * grad
            w_new = (1 - a_) * state.weights[0] + s
            return MemoryState(weights=[w_new], momentum=[s])

        else:
            grads = self._autograd_grads(keys, values)

            new_weights, new_momentum = [], []
            for g, w, m in zip(grads, state.weights, state.momentum):
                s     = e_ * m - h_ * g
                w_new = (1 - a_) * w + s
                new_weights.append(w_new)
                new_momentum.append(s)
            return MemoryState(weights=new_weights, momentum=new_momentum)

    def _autograd_grads(
        self,
        keys:   torch.Tensor,   # (B, S, D)
        values: torch.Tensor,   # (B, S, D)
    ) -> list[torch.Tensor]:
        """One autograd pass over the full sequence."""
        with torch.enable_grad():
            for p in self.memory.parameters():
                p.requires_grad_(True)

            loss = self.memory.associative_loss(keys.detach(), values.detach())

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

    # ------------------------------------------------------------------
    # Read: full sequence at once (no chunking)
    # ------------------------------------------------------------------
    def _read(
        self,
        queries: torch.Tensor,   # (B, S, D)
        state:   MemoryState,
    ) -> torch.Tensor:
        self.memory.set_weights(state.weights)
        return self.memory(queries)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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

        # Projections
        k = F.normalize(F.silu(self.proj_k(x)), p=2, dim=-1)
        v = F.silu(self.proj_v(x))
        q = F.normalize(F.silu(self.proj_q(x)), p=2, dim=-1)

        # Gates
        alpha = self.gate_alpha(x)
        theta = self.gate_theta(x)
        eta   = self.gate_eta(x)

        # Single write pass over the full sequence
        state = self._write(k, v, alpha, theta, eta, state)

        # Single read pass over the full sequence
        output = self._read(q, state)

        output = self.norm_out(self.proj_out(output))
        return output, state.detach_to(device)