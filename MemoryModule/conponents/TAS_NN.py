# coding=utf-8
"""
Trajectory-Aware Surprise (TAS) — paper-faithful implementation.

Equations (from slides):

  S_t^TAS = θ_t · m_t · ∇L(x_t; M_{t-1})          [TAS signal]
  S_t     = η_t · S_{t-1} − S_t^TAS                 [persistent surprise state]
  M_t     = (1 − α_t) · M_{t-1} + S_t               [memory update]

  m_t     = σ(MetaMLP(τ_t))                          [trajectory modulation gate]
  τ_t     = [h_t, Δh_t, ||∇L_t||, ΔL_t, S_{t-1}]  [trajectory features]

Trajectory feature definitions (window size C):
  Δh_t     = (h_t − h_{t−C+1}) / C                  [hidden drift]
  ||∇L_t|| = mean(∇L_{t−C+1} ... t)                 [gradient trend]
  ΔL_t     = mean(L_{t−C+1...t−1}) − L_t            [loss trend]

This module is a drop-in replacement for NeuralLongTermMemory.
The existing MemoryMLP and MemoryState are reused unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TitansConfig
from .memory import MemoryMLP, MemoryState, RMSNorm


# ── TASState ───

@dataclass
class TASState:
    """
    Carries everything that must persist across chunks / time-steps.

    memory   : weight + momentum of the MemoryMLP  (existing MemoryState)
    surprise : S_{t-1} — one tensor per weight layer, same shape as weights
    h_window : ring buffer of hidden states, shape (C, D)
    grad_window  : ring buffer of per-step gradient norms, shape (C,)
    loss_window  : ring buffer of per-step scalar losses,   shape (C,)
    step     : integer cursor into the ring buffers
    """
    memory:      MemoryState
    surprise:    list[torch.Tensor]          # S_{t-1}, one per layer
    h_window:    torch.Tensor                # (C, D)
    grad_window: torch.Tensor                # (C,)
    loss_window: torch.Tensor                # (C,)
    step:        int = 0

    def detach(self) -> TASState:
        return TASState(
            memory      = self.memory.detach(),
            surprise    = [s.detach() for s in self.surprise],
            h_window    = self.h_window.detach(),
            grad_window = self.grad_window.detach(),
            loss_window = self.loss_window.detach(),
            step        = self.step,
        )

    def to(self, device: torch.device) -> TASState:
        return TASState(
            memory      = self.memory.to(device),
            surprise    = [s.to(device) for s in self.surprise],
            h_window    = self.h_window.to(device),
            grad_window = self.grad_window.to(device),
            loss_window = self.loss_window.to(device),
            step        = self.step,
        )

    def detach_to(self, device: torch.device) -> TASState:
        return TASState(
            memory      = self.memory.detach_to(device),
            surprise    = [s.detach().to(device) for s in self.surprise],
            h_window    = self.h_window.detach().to(device),
            grad_window = self.grad_window.detach().to(device),
            loss_window = self.loss_window.detach().to(device),
            step        = self.step,
        )


# MetaMLP (hyper-network)

class MetaMLP(nn.Module):
    """
    Lightweight hyper-network that maps trajectory features τ_t → m_t.

    Input dimension  : tau_dim  (= 4*D + 1, see _build_tau)
    Output dimension : gate_dim (= D, element-wise gate over the gradient)

    m_t = σ(MetaMLP(τ_t))
    """

    def __init__(self, tau_dim: int, gate_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tau_dim,    hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, gate_dim,   bias=True),
        )
        self._init()

    def _init(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (tau_dim,)  →  gate: (gate_dim,) in (0,1)"""
        return torch.sigmoid(self.net(tau))


# ── TAS NeuralLongTermMemory ──────────────────────────────────────────────────

class TASNeuralLongTermMemory(nn.Module):
    """
    Neural Long-Term Memory with Trajectory-Aware Surprise.

    Replaces NeuralLongTermMemory; the external interface is identical:
        output, new_state = module(x, encoder_hidden_states, state)

    `state` is now a TASState instead of a MemoryState.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config     = config
        self.dim        = config.dim
        self.chunk_size = getattr(config, "chunk_size", 64)
        self.window_C   = getattr(config, "tas_window", 16)   # C in slides

        # ── Projections (identical to original) ──────────────────────
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        # ── Memory MLP (identical to original) ───────────────────────
        self.memory = MemoryMLP(config)

        # ── Learnable scalar gates (α, θ, η) — all token-level ───────
        # α_t : forgetting coefficient    — (D,) sigmoid output
        # θ_t : TAS scaling factor        — (D,) sigmoid output
        # η_t : surprise retention coeff  — (D,) sigmoid output
        self.gate_alpha = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())
        self.gate_theta = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())
        self.gate_eta   = nn.Sequential(nn.Linear(config.dim, config.dim, bias=False), nn.Sigmoid())

        # ── Trajectory Modulation Gate (MetaMLP / HyperNetwork) ──────
        # τ_t = [h_t (D), Δh_t (D), ||∇L_t|| (D), ΔL_t (1), S_{t-1} (D)]
        # dim  =  D  +   D    +     D          +  1   +    D   = 4D+1
        tau_dim  = 4 * config.dim + 1
        gate_dim = config.dim
        meta_hidden = getattr(config, "meta_mlp_hidden", 128)
        self.meta_mlp = MetaMLP(tau_dim, gate_dim, meta_hidden)

        # ── Output projection ─────────────────────────────────────────
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)
        self.norm_out = RMSNorm(config.dim)

        self._init_weights()

    def _init_weights(self) -> None:
        std = getattr(self.config, "init_std", 0.02)
        for m in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            nn.init.normal_(m.weight, std=std)

    # ── State initialisation ──────────────────────────────────────────────────

    def init_state(self, device: torch.device) -> TASState:
        weights  = [w.clone().to(device) for w in self.memory.get_weights()]
        momentum = [torch.zeros_like(w) for w in weights]
        surprise = [torch.zeros_like(w) for w in weights]
        return TASState(
            memory      = MemoryState(weights=weights, momentum=momentum),
            surprise    = surprise,
            h_window    = torch.zeros(self.window_C, self.dim, device=device),
            grad_window = torch.zeros(self.window_C,           device=device),
            loss_window = torch.zeros(self.window_C,           device=device),
            step        = 0,
        )

    # ── Trajectory feature computation ───────────────────────────────────────

    def _compute_tau(
        self,
        h_t:         torch.Tensor,   # (D,)  current hidden state (mean over B,T)
        grad_norm_t: torch.Tensor,   # scalar, ||∇L_t|| for this chunk
        loss_t:      torch.Tensor,   # scalar, L_t for this chunk
        state:       TASState,
    ) -> torch.Tensor:
        """
        Build τ_t = [h_t, Δh_t, ||∇L_t||_trend, ΔL_t, S_{t-1}]
        and update the ring buffers inside state (in-place on the tensors).

        Returns τ_t as a 1-D tensor of shape (4D+1,).
        """
        C    = self.window_C
        step = state.step % C

        # ── hidden drift:  Δh_t = (h_t − h_{t−C+1}) / C  ────────────
        h_oldest = state.h_window[step].clone()        # position about to be overwritten = oldest
        delta_h  = (h_t - h_oldest) / C

        # ── gradient trend: mean of gradient norms over last C steps ──
        # We overwrite *after* computing the mean so the current step counts
        grad_trend = state.grad_window.mean()           # scalar before update

        # ── loss trend: mean(L_{t-C+1..t-1}) − L_t ───────────────────
        # The window contains the last C losses; oldest will be replaced.
        loss_trend = state.loss_window.mean() - loss_t  # scalar

        # ── S_{t-1}: average across layers (flattened mean as a D-vec) ─
        # We represent momentum surprise S_{t-1} as its mean per-layer norm
        # projected onto a D-dimensional summary via mean-pooling layers.
        s_prev_scalar = torch.stack(
            [s.norm() for s in state.surprise]
        ).mean().unsqueeze(0)                          # (1,)

        # Use first surprise layer flattened if it matches D, else norm
        if state.surprise[0].shape == (self.dim, self.dim):
            # weight matrix: take row-wise norm summary → (D,)
            s_prev_vec = state.surprise[0].norm(dim=1)
        else:
            s_prev_vec = state.surprise[0].reshape(-1)[:self.dim]

        tau = torch.cat([
            h_t,                                        # (D,)
            delta_h,                                    # (D,)
            grad_norm_t.expand(self.dim),               # (D,)  broadcast scalar
            loss_trend.unsqueeze(0),                    # (1,)
            s_prev_vec,                                 # (D,)
        ], dim=0)                                       # (4D+1,)

        # ── Update ring buffers ───────────────────────────────────────
        state.h_window[step]    = h_t.detach()
        state.grad_window[step] = grad_norm_t.detach()
        state.loss_window[step] = loss_t.detach()
        state.step             += 1

        return tau

    # ── Per-chunk write with TAS ──────────────────────────────────────────────

    def _write_chunk_tas(
        self,
        keys:   torch.Tensor,   # (B, C_w, D)
        values: torch.Tensor,   # (B, C_w, D)
        alpha:  torch.Tensor,   # (B, C_q, D)  forgetting gate
        theta:  torch.Tensor,   # (B, C_q, D)  TAS scaling gate
        eta:    torch.Tensor,   # (B, C_q, D)  surprise retention gate
        h_t:    torch.Tensor,   # (D,)          mean hidden for τ
        state:  TASState,
    ) -> TASState:
        """
        Full TAS update for one chunk:

            g_t     = ∇L(x_t; M_{t-1})          [gradient of memory loss]
            m_t     = σ(MetaMLP(τ_t))            [trajectory gate]
            S_t^TAS = θ_t · m_t · g_t           [TAS signal — eq. from slide 3]
            S_t     = η_t · S_{t-1} − S_t^TAS   [persistent surprise — slide 2]
            M_t     = (1 − α_t) · M_{t-1} + S_t [memory update — slide 1]
        """
        self.memory.set_weights(state.memory.weights)

        # ── Gate scalars: mean over batch + query-chunk tokens ────────
        a = alpha.mean(dim=(0, 1))   # (D,)  α_t
        th = theta.mean(dim=(0, 1))  # (D,)  θ_t
        et = eta.mean(dim=(0, 1))    # (D,)  η_t

        # ── Compute raw gradients g_t = ∇_W L ────────────────────────
        grads = self._compute_chunk_grads(keys, values)

        # ── Compute trajectory features τ_t ──────────────────────────
        # grad_norm_t : mean ||∇|| across layers
        grad_norm_t = torch.stack(
            [g.norm() for g in grads]
        ).mean()

        # loss_t : reconstruction loss for this chunk
        with torch.no_grad():
            loss_t = self.memory.associative_loss(
                keys.detach().reshape(-1, self.dim),
                values.detach().reshape(-1, self.dim),
            )

        tau = self._compute_tau(h_t, grad_norm_t, loss_t, state)

        # ── Trajectory modulation gate: m_t = σ(MetaMLP(τ_t)) ────────
        m_gate = self.meta_mlp(tau)   # (D,)

        # ── Per-layer TAS update ──────────────────────────────────────
        new_weights  = []
        new_surprise = []

        for g, w, s_prev in zip(grads, state.memory.weights, state.surprise):
            # Broadcast gate tensors to weight shape (D_out, D_in)
            # θ_t and m_t are (D,) — treat as per-output-dimension scale
            a_   = a.unsqueeze(-1)    # (D, 1)
            th_  = th.unsqueeze(-1)   # (D, 1)
            et_  = et.unsqueeze(-1)   # (D, 1)
            m_   = m_gate.unsqueeze(-1) if g.shape[0] == self.dim \
                   else m_gate.unsqueeze(0)

            # S_t^TAS = θ_t · m_t · g_t
            s_tas = th_ * m_ * g

            # S_t = η_t · S_{t-1} − S_t^TAS
            s_new = et_ * s_prev - s_tas

            # M_t = (1 − α_t) · M_{t-1} + S_t
            w_new = (1 - a_) * w + s_new

            new_weights.append(w_new)
            new_surprise.append(s_new)

        return TASState(
            memory      = MemoryState(
                weights  = new_weights,
                momentum = state.memory.momentum,   # momentum not used in TAS eqs
            ),
            surprise    = new_surprise,
            h_window    = state.h_window,
            grad_window = state.grad_window,
            loss_window = state.loss_window,
            step        = state.step,
        )

    #Gradient computation (batched, identical logic to base class) 

    def _compute_chunk_grads(
        self,
        keys:   torch.Tensor,   # (B, C, D)
        values: torch.Tensor,   # (B, C, D)
    ) -> list[torch.Tensor]:
        """Compute ∇_W MSE(M(k), v) for the full chunk in one autograd pass."""
        if len(self.memory.layers) == 1:
            W   = self.memory.layers[0].weight              # (D, D)
            Wk  = keys @ W.t()                              # (B, C, D)
            err = Wk - values                               # (B, C, D)
            N   = keys.shape[0] * keys.shape[1]
            g   = torch.einsum("bco,bci->oi", err, keys) / N
            return [g]
        else:
            with torch.enable_grad():
                for p in self.memory.parameters():
                    p.requires_grad_(True)
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
                for g, p in zip(grads, self.memory.parameters())
            ]

    # Read (unchanged from original)

    def _read_chunk(
        self,
        queries: torch.Tensor,
        state:   TASState,
    ) -> torch.Tensor:
        self.memory.set_weights(state.memory.weights)
        return self.memory(queries)

    # Forward 

    def forward(
        self,
        x:                     torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        state:                 TASState | None      = None,
    ) -> tuple[torch.Tensor, TASState]:
        B, S, D = x.shape
        device  = x.device

        if state is None:
            state = self.init_state(device)
        if state.memory.weights[0].device != device:
            state = state.to(device)

        # Projections 
        k = F.normalize(F.silu(self.proj_k(x)), p=2, dim=-1)
        v = F.silu(self.proj_v(x))
        q = F.normalize(F.silu(self.proj_q(x)), p=2, dim=-1)

        # Gates 
        alpha = self.gate_alpha(x)   # (B, S, D)  α_t — forgetting
        theta = self.gate_theta(x)   # (B, S, D)  θ_t — TAS scaling
        eta   = self.gate_eta(x)     # (B, S, D)  η_t — surprise retention

        # Padding 
        C     = self.chunk_size
        pad_w = (-k.shape[1]) % C
        if pad_w:
            k = F.pad(k, (0, 0, 0, pad_w))
            v = F.pad(v, (0, 0, 0, pad_w))

        pad_r = (-S) % C
        if pad_r:
            q     = F.pad(q,     (0, 0, 0, pad_r))
            alpha = F.pad(alpha, (0, 0, 0, pad_r))
            theta = F.pad(theta, (0, 0, 0, pad_r))
            eta   = F.pad(eta,   (0, 0, 0, pad_r))

        n_write = k.shape[1] // C
        n_read  = q.shape[1] // C

        # Write pass 
        for c in range(n_write):
            wsl   = slice(c * C, (c + 1) * C)
            q_idx = min(c, n_read - 1)
            qsl   = slice(q_idx * C, (q_idx + 1) * C)

            # Mean hidden state for this chunk → trajectory feature h_t
            h_t = x[:, c * C : min((c + 1) * C, S)].mean(dim=(0, 1))  # (D,)

            state = self._write_chunk_tas(
                k[:, wsl], v[:, wsl],
                alpha[:, qsl], theta[:, qsl], eta[:, qsl],
                h_t,
                state,
            )

        # Read pass
        outputs = [
            self._read_chunk(q[:, c * C : (c + 1) * C], state)
            for c in range(n_read)
        ]
        output = torch.cat(outputs, dim=1)[:, :S]

        output = self.norm_out(self.proj_out(output))
        return output, state.detach_to(device)