from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TitansConfig
from .memory import MemoryMLP, MemoryState, RMSNorm



# TAS STATE
@dataclass
class TASState:
    """
    Persistent state for TAS-based long-term memory.

    memory      : MemoryState containing memory weights + momentum
    surprise    : persistent surprise tensors, one per memory parameter
    h_window    : ring buffer of hidden summaries, shape (C, D)
    grad_window : ring buffer of scalar gradient norms, shape (C,)
    loss_window : ring buffer of scalar losses, shape (C,)
    step        : current step index for ring-buffer overwrite
    """
    memory: MemoryState
    surprise: list[torch.Tensor]
    h_window: torch.Tensor
    grad_window: torch.Tensor
    loss_window: torch.Tensor
    step: int = 0

    def detach(self) -> "TASState":
        return TASState(
            memory=self.memory.detach(),
            surprise=[s.detach() for s in self.surprise],
            h_window=self.h_window.detach(),
            grad_window=self.grad_window.detach(),
            loss_window=self.loss_window.detach(),
            step=self.step,
        )

    def to(self, device: torch.device) -> "TASState":
        return TASState(
            memory=self.memory.to(device),
            surprise=[s.to(device) for s in self.surprise],
            h_window=self.h_window.to(device),
            grad_window=self.grad_window.to(device),
            loss_window=self.loss_window.to(device),
            step=self.step,
        )

    def detach_to(self, device: torch.device) -> "TASState":
        return TASState(
            memory=self.memory.detach_to(device),
            surprise=[s.detach().to(device) for s in self.surprise],
            h_window=self.h_window.detach().to(device),
            grad_window=self.grad_window.detach().to(device),
            loss_window=self.loss_window.detach().to(device),
            step=self.step,
        )



# META MLP


class MetaMLP(nn.Module):
    """
    Meta-network for trajectory modulation gate:

        m_t = σ(MetaMLP(τ_t))
    """

    def __init__(self, tau_dim: int, gate_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tau_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, gate_dim),
        )
        self._init()

    def _init(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(tau))



# TAS MEMORY MODULE


class TASNeuralLongTermMemory(nn.Module):
    """
    Thesis-aligned TAS implementation.

    Core equations:

        S_t^TAS = θ_t · m_t · ∇L(x_t; M_{t-1})
        S_t     = η_t · S_{t-1} − S_t^TAS
        M_t     = (1 − α_t) · M_{t-1} + S_t

        m_t     = σ(MetaMLP(τ_t))

    Trajectory feature vector:

        τ_t = [h_t, Δh_t, ||∇L_t||_trend, ΔL_t, S_{t-1}^{summary}]

    Here:
      - h_t                 : D-dimensional hidden summary
      - Δh_t                : D-dimensional hidden drift
      - ||∇L_t||_trend      : scalar windowed mean gradient norm
      - ΔL_t                : scalar loss trend
      - S_{t-1}^{summary}   : D-dimensional summary of persistent surprise
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.chunk_size = getattr(config, "chunk_size", 64)
        self.window_C = getattr(config, "tas_window", 16)

        # Projections
        self.proj_k = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_q = nn.Linear(self.dim, self.dim, bias=False)

        # Memory backbone
        self.memory = MemoryMLP(config)

        # Token-level gates
        self.gate_alpha = nn.Sequential(nn.Linear(self.dim, self.dim, bias=False), nn.Sigmoid())
        self.gate_theta = nn.Sequential(nn.Linear(self.dim, self.dim, bias=False), nn.Sigmoid())
        self.gate_eta = nn.Sequential(nn.Linear(self.dim, self.dim, bias=False), nn.Sigmoid())

        # τ_t = [h_t (D), Δh_t (D), grad_trend (1), loss_trend (1), S_prev_summary (D)]
        tau_dim = 3 * self.dim + 2
        gate_dim = self.dim
        meta_hidden = getattr(config, "meta_mlp_hidden", 128)
        self.meta_mlp = MetaMLP(tau_dim=tau_dim, gate_dim=gate_dim, hidden_dim=meta_hidden)

        self.proj_out = nn.Linear(self.dim, self.dim, bias=False)
        self.norm_out = RMSNorm(self.dim)

        self._init_weights()

    def _init_weights(self) -> None:
        std = getattr(self.config, "init_std", 0.02)
        for m in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            nn.init.normal_(m.weight, std=std)

    
    # State init
    def init_state(self, device: torch.device) -> TASState:
        weights = [w.clone().to(device) for w in self.memory.get_weights()]
        momentum = [torch.zeros_like(w) for w in weights]
        surprise = [torch.zeros_like(w) for w in weights]

        return TASState(
            memory=MemoryState(weights=weights, momentum=momentum),
            surprise=surprise,
            h_window=torch.zeros(self.window_C, self.dim, device=device),
            grad_window=torch.zeros(self.window_C, device=device),
            loss_window=torch.zeros(self.window_C, device=device),
            step=0,
        )

    
    # Helpers
    def _summarize_surprise(self, surprise_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Create a D-dimensional summary of persistent surprise across all layers.

        Strategy:
          - if tensor is matrix-like, use row norms
          - if tensor is vector-like, use itself
          - resize/pad to D and average over layers
        """
        summaries = []
        for s in surprise_list:
            if s.ndim == 2:
                vec = s.norm(dim=1)  # shape: (out_dim,)
            elif s.ndim == 1:
                vec = s
            else:
                vec = s.reshape(-1)

            if vec.numel() >= self.dim:
                vec = vec[: self.dim]
            else:
                vec = F.pad(vec, (0, self.dim - vec.numel()))

            summaries.append(vec)

        return torch.stack(summaries, dim=0).mean(dim=0)  # (D,)

    def _broadcast_gate_to_param(self, gate_vec: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        Map a D-dimensional gate vector to the shape of a parameter tensor.

        For 2D weights: gate is applied row-wise (per output feature).
        For 1D params  : gate is resized to that length.
        For higher dims: reshape-safe fallback using first dimension.
        """
        target0 = param.shape[0]

        if gate_vec.numel() == target0:
            g = gate_vec
        elif gate_vec.numel() > target0:
            g = gate_vec[:target0]
        else:
            g = F.pad(gate_vec, (0, target0 - gate_vec.numel()))

        if param.ndim == 1:
            return g
        elif param.ndim == 2:
            return g.unsqueeze(-1).expand_as(param)
        else:
            shape = [target0] + [1] * (param.ndim - 1)
            return g.view(*shape).expand_as(param)

    
    # Trajectory feature computation
    def _compute_tau(
        self,
        h_t: torch.Tensor,        # (D,)
        grad_norm_t: torch.Tensor,  # scalar
        loss_t: torch.Tensor,       # scalar
        state: TASState,
    ) -> torch.Tensor:
        """
        Build τ_t and update the ring buffers.

        Thesis-aligned feature structure:
            τ_t = [h_t, Δh_t, grad_trend, loss_trend, S_prev_summary]
        """
        C = self.window_C
        step = state.step % C

        # oldest element that will be overwritten
        h_oldest = state.h_window[step].clone()

        # hidden drift
        delta_h = (h_t - h_oldest) / C

        # form temporary updated windows so current step is included
        grad_window_new = state.grad_window.clone()
        grad_window_new[step] = grad_norm_t.detach()
        grad_trend = grad_window_new.mean()  # scalar

        loss_window_new = state.loss_window.clone()
        loss_window_new[step] = loss_t.detach()

        # mean(previous/current window except current target tendency) - current loss
        loss_trend = loss_window_new.mean() - loss_t  # scalar

        s_prev_summary = self._summarize_surprise(state.surprise)  # (D,)

        tau = torch.cat(
            [
                h_t,                         # (D,)
                delta_h,                     # (D,)
                grad_trend.unsqueeze(0),     # (1,)
                loss_trend.unsqueeze(0),     # (1,)
                s_prev_summary,              # (D,)
            ],
            dim=0,
        )  # (3D + 2,)

        # update buffers in-place
        state.h_window[step] = h_t.detach()
        state.grad_window[step] = grad_norm_t.detach()
        state.loss_window[step] = loss_t.detach()
        state.step += 1

        return tau

    
    # Gradient computation
    def _compute_chunk_grads(
        self,
        keys: torch.Tensor,    # (B, C, D)
        values: torch.Tensor,  # (B, C, D)
    ) -> list[torch.Tensor]:
        """
        Compute ∇_W associative loss for the full chunk.
        """
        with torch.enable_grad():
            params = list(self.memory.parameters())
            for p in params:
                p.requires_grad_(True)

            loss = self.memory.associative_loss(
                keys.detach(),
                values.detach(),
            )

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
            for g, p in zip(grads, params)
        ]

    
    # TAS write update
    def _write_chunk_tas(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        alpha: torch.Tensor,
        theta: torch.Tensor,
        eta: torch.Tensor,
        h_t: torch.Tensor,
        state: TASState,
    ) -> TASState:
        """
        Full TAS update for one chunk.
        """
        self.memory.set_weights(state.memory.weights)

        # chunk-level gate summaries
        a = alpha.mean(dim=(0, 1))    # (D,)
        th = theta.mean(dim=(0, 1))   # (D,)
        et = eta.mean(dim=(0, 1))     # (D,)

        grads = self._compute_chunk_grads(keys, values)

        with torch.no_grad():
            grad_norm_t = torch.stack([g.norm() for g in grads]).mean()
            loss_t = self.memory.associative_loss(
                keys.detach().reshape(-1, self.dim),
                values.detach().reshape(-1, self.dim),
            )

        tau = self._compute_tau(h_t, grad_norm_t, loss_t, state)

        # m_t = σ(MetaMLP(τ_t))
        m_gate = self.meta_mlp(tau)   # (D,)

        new_weights = []
        new_surprise = []

        for g, w, s_prev in zip(grads, state.memory.weights, state.surprise):
            a_map = self._broadcast_gate_to_param(a, w)
            th_map = self._broadcast_gate_to_param(th, w)
            et_map = self._broadcast_gate_to_param(et, w)
            m_map = self._broadcast_gate_to_param(m_gate, w)

            # S_t^TAS = θ_t · m_t · ∇L
            s_tas = th_map * m_map * g

            # S_t = η_t · S_{t-1} − S_t^TAS
            s_new = et_map * s_prev - s_tas

            # M_t = (1 − α_t) · M_{t-1} + S_t
            w_new = (1.0 - a_map) * w + s_new

            new_weights.append(w_new)
            new_surprise.append(s_new)

        return TASState(
            memory=MemoryState(
                weights=new_weights,
                momentum=state.memory.momentum,  # retained for compatibility
            ),
            surprise=new_surprise,
            h_window=state.h_window,
            grad_window=state.grad_window,
            loss_window=state.loss_window,
            step=state.step,
        )

    
    # Read
    def _read_chunk(
        self,
        queries: torch.Tensor,
        state: TASState,
    ) -> torch.Tensor:
        self.memory.set_weights(state.memory.weights)
        return self.memory(queries)

    
    # Forward
    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        state: TASState | None = None,
    ) -> tuple[torch.Tensor, TASState]:
        B, S, D = x.shape
        device = x.device

        if state is None:
            state = self.init_state(device)
        if state.memory.weights[0].device != device:
            state = state.to(device)

        # projections
        k = F.normalize(F.silu(self.proj_k(x)), p=2, dim=-1)
        v = F.silu(self.proj_v(x))
        q = F.normalize(F.silu(self.proj_q(x)), p=2, dim=-1)

        # token-level gates
        alpha = self.gate_alpha(x)
        theta = self.gate_theta(x)
        eta = self.gate_eta(x)

        # pad write chunks
        C = self.chunk_size
        pad_w = (-k.shape[1]) % C
        if pad_w:
            k = F.pad(k, (0, 0, 0, pad_w))
            v = F.pad(v, (0, 0, 0, pad_w))

        # pad read chunks
        pad_r = (-S) % C
        if pad_r:
            q = F.pad(q, (0, 0, 0, pad_r))
            alpha = F.pad(alpha, (0, 0, 0, pad_r))
            theta = F.pad(theta, (0, 0, 0, pad_r))
            eta = F.pad(eta, (0, 0, 0, pad_r))

        n_write = k.shape[1] // C
        n_read = q.shape[1] // C

        # write pass
        for c in range(n_write):
            wsl = slice(c * C, (c + 1) * C)
            q_idx = min(c, n_read - 1)
            qsl = slice(q_idx * C, (q_idx + 1) * C)

            h_t = x[:, c * C: min((c + 1) * C, S)].mean(dim=(0, 1))  # (D,)

            state = self._write_chunk_tas(
                k[:, wsl],
                v[:, wsl],
                alpha[:, qsl],
                theta[:, qsl],
                eta[:, qsl],
                h_t,
                state,
            )

        # read pass
        outputs = []
        for c in range(n_read):
            rsl = slice(c * C, (c + 1) * C)
            outputs.append(self._read_chunk(q[:, rsl], state))

        output = torch.cat(outputs, dim=1)[:, :S]
        output = self.norm_out(self.proj_out(output))

        return output, state.detach_to(device)