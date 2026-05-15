# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.func import functional_call
# class ContextNeuralMemory(nn.Module):
#     """
#     Neural memory module with online/continual learning capability.
    
#     The module learns to transform normalized keys → values
#     via surprise-modulated parameter updates (Hebbian-like online update).
    
#     K and V are usually kept fixed (feature extractors),
#     while the intermediate layers are plastic / online-learnable.
#     """
#     def __init__(
#         self,
#         emb_dim: int = 64,
#         n_layers: int = 3,
#         hidden_dim: int = 128,
#         surprise_init_scale: float = 0.005,
#         default_alpha: float = 0.995,   # retention / momentum term
#         default_theta: float = 0.05,    # learning rate scale
#         default_eta: float = 0.97,      # surprise decay
#     ):
#         super().__init__()
#         self.emb_dim = emb_dim

#         # ── Fixed key & value projections (usually not updated online) ──
#         self.K = nn.Linear(emb_dim, emb_dim, bias=False)
#         self.V = nn.Linear(emb_dim, emb_dim, bias=False)

#         nn.init.xavier_uniform_(self.K.weight)
#         nn.init.xavier_uniform_(self.V.weight)

#         # ── Learnable transformation layers ─────────────────────────────
#         layers = []
#         current_dim = emb_dim
#         for i in range(n_layers):
#             next_dim = hidden_dim if i < n_layers - 1 else emb_dim
#             layers.append(nn.Linear(current_dim, next_dim))
#             if i < n_layers - 1:
#                 layers.append(nn.SiLU())
#             current_dim = next_dim

#         self.layers = nn.Sequential(*layers)

#         # ── Surprise memory (plain dict – allows dotted names like '0.weight') ──
#         self.surprise = {}
#         for name, param in self.layers.named_parameters():
#             if param.requires_grad:
#                 buf = torch.zeros_like(param.data)
#                 buf *= surprise_init_scale
#                 self.surprise[name] = buf

#         # Hyperparameters as buffers (can be overridden per call or scheduled)
#         self.register_buffer("alpha", torch.tensor(default_alpha, dtype=torch.float32))
#         self.register_buffer("theta", torch.tensor(default_theta, dtype=torch.float32))
#         self.register_buffer("eta",   torch.tensor(default_eta,   dtype=torch.float32))

#     def retrieve(self, x):
#         return functional_call(self, dict(self.named_parameters()), x)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Standard inference / retrieval path.
#         Returns the transformed representation.
#         """
#         z = x
#         keys = F.normalize(F.silu(self.K(z)), p=2, dim=-1)
#         out = self.layers(keys)
#         return out
    

#     @torch.no_grad()
#     def update(
#         self,
#         x: torch.Tensor,
#         alpha: float | torch.Tensor | None = None,
#         theta: float | torch.Tensor | None = None,
#         eta:   float | torch.Tensor | None = None,
#         grad_clip: float = 5.0,
#     ) -> tuple[float, dict]:
#         # # Update hyperparameters if provided
#         # if alpha is not None:
#         #     self.alpha.copy_(torch.as_tensor(alpha, device=self.alpha.device))
#         # if theta is not None:
#         #     self.theta.copy_(torch.as_tensor(theta, device=self.theta.device))
#         # if eta is not None:
#         #     self.eta.copy_(torch.as_tensor(eta, device=self.eta.device))

#         z = x.detach()

#         # ── Forward ────────────────────────────────────────
#         keys = F.normalize(F.silu(self.K(z)), p=2, dim=-1)
#         pred = self.layers(keys)
#         target = F.silu(self.V(z))
#         loss = F.mse_loss(pred, target)

#         # ── Only parameters that actually require grad ─────
#         params = []
#         names  = []
#         for name, param in self.layers.named_parameters():
#             if param.requires_grad:
#                 params.append(param)
#                 names.append(name)

#         if not params:
#             print("Warning: No parameters with requires_grad=True in self.layers")
#             return loss.item(), {}

#         grads = torch.autograd.grad(
#             loss,
#             params,
#             retain_graph=False,
#             allow_unused=False   # now safe — we filtered
#         )

#         updated_params = {}
#         loss_value = loss.item()

#         for name, param, grad in zip(names, params, grads):
#             if grad is None:
#                 # This should no longer happen with allow_unused=False + filtering
#                 continue

#             grad = grad.clamp_(-grad_clip, grad_clip)

#             if name not in self.surprise:
#                 self.surprise[name] = torch.zeros_like(param.data)

#             surprise = self.surprise[name]
#             surprise.mul_(self.eta).sub_(self.theta * grad)

#             new_value = self.alpha * param.data + surprise
#             param.data.copy_(new_value.detach())   # .detach() optional but safer

#             updated_params[name] = new_value.detach().clone()

#         return loss_value, updated_params

#     def extra_repr(self) -> str:
#         return (f"emb_dim={self.emb_dim}, "
#                 f"n_layers={len(self.layers)//2}, "
#                 f"hidden={self.layers[0].out_features if hasattr(self.layers[0], 'out_features') else '?'}, "
#                 f"α={self.alpha.item():.4f}, θ={self.theta.item():.4f}, η={self.eta.item():.4f}")

#     def reset_surprise(self):
#         """Reset surprise terms to zero (e.g. after task switch)"""
#         for name in self.surprise:
#             self.surprise[name].zero_()


import torch
import torch.nn as nn
from torch.func import functional_call
import torch.nn.functional as F
class ContextNeuralMemory(nn.Module):
    def __init__(self, emb_dim = 16, n_layers= 2, hidden_dim = 32):
        super().__init__()
        # Define the layers of the network
        self.layers = None
        if n_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(emb_dim, emb_dim)])
        else:
            self.layers = nn.ModuleList([])
            self.layers.append(nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.SiLU()
            ))
            for k in range(n_layers - 2):
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU()
                ))
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, emb_dim)
            ))

        # Mapping to keys
        self.K = nn.Linear(emb_dim, emb_dim, bias = False)

        # Mapping to values
        self.V = nn.Linear(emb_dim, emb_dim, bias = False)

        torch.nn.init.xavier_uniform_(self.K.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)

        self.silu = nn.SiLU()
        self.surprise = {}
    
    def retrieve(self, x):
        return functional_call(self, dict(self.named_parameters()), x)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def update(self, x, alpha, theta, eta):
        """
        Performs a test-time / online / surprise-based parameter update.
        
        Args:
            x: input tensor
            alpha: step-size / learning rate factor (scalar or tensor)
            theta: decay / momentum-like coefficient for surprise
            eta: decay factor for surprise accumulator
        
        Returns:
            loss_value (float): the computed MSE loss
            updated_params (dict): name → updated parameter tensor (detached)
        """
        # Forward pass (K and V are frozen w.r.t. this update)
        # print("x before detach", x)
        with torch.enable_grad():
            z = x.detach().clone().requires_grad_(True)

            params = [p for p in self.parameters() if p.requires_grad]

            keys = F.silu(self.K(z))
            vals = F.silu(self.V(z))
            keys = F.normalize(keys, p=2, dim=-1)

            for layer in self.layers:
                keys = layer(keys)
                keys = F.layer_norm(keys, normalized_shape=keys.shape[1:])

            mse_loss = ((keys - vals) ** 2).mean()

            aux_loss = 1e-7 * sum(p.abs().sum() for p in params)
            total_loss = mse_loss + aux_loss

            grads = torch.autograd.grad(
                total_loss,
                params,
                retain_graph=False,
                allow_unused=False,
            )
            
            # Prepare clipped gradients (replace None → zero, though unlikely now)
            clipped_grads = []
            for grad, param in zip(grads, self.parameters()):
                if grad is None:
                    # This branch should almost never be reached thanks to aux_loss
                    grad = torch.zeros_like(param)
                else:
                    # You can adjust clipping range as needed
                    grad = torch.clamp(grad, min=-1e3, max=1e3)
                clipped_grads.append(grad)

            # Extract scalar values for update coefficients
            alpha_g = alpha.mean().detach()
            theta_g = theta.mean().detach()
            eta_g = eta.mean().detach()

            # Dictionary to store updated parameter values (for logging / inspection)
            updated_params = {}

            # Apply surprise-based update
            for (name, param), grad in zip(self.named_parameters(), clipped_grads):
                # Initialize surprise buffer if not present
                if name not in self.surprise:
                    self.surprise[name] = torch.zeros_like(param)

                # Update surprise accumulator
                # surprise ← eta * surprise - theta * grad
                self.surprise[name] = eta_g * self.surprise[name] - theta_g * grad

                if name.startswith("K.") or name.startswith("V."):
                    # K and V are **not** adapted via surprise
                    updated = param.data
                else:
                    # Surprise-driven update for other parameters
                    updated = alpha_g * param.data + self.surprise[name]

                # Apply the update in-place
                param.data.copy_(updated)

                # Store for return value
                updated_params[name] = updated.detach()  # detached copy

        return mse_loss.item(), updated_params
    

    # def update(self, x, alpha, theta, eta):
    #     z = x.detach()
    #     # Evaluate the corresponding keys and values
    #     keys = normalize(self.silu(self.K(z)))
    #     vals = self.silu(self.V(z))
    #     # Propagate the keys through the model
    #     for layer in self.layers:
    #         keys = layer(keys)

  
    #     # Calculate the loss || M(keys) - vals ||_2 ^2
    #     loss = ((keys - vals) ** 2).mean(axis=0).sum()

    #     # print("loss", loss)

    #     # Compute gradients of aux loss w.r.t. NMM's parameters
    #     grads = torch.autograd.grad(loss, self.parameters())

    #     print("garads", grads)

    #     # Update the surprise dictionary and the parameters of the network
    #     updated_params = {}

    #     # print("key", keys.shape)
    #     # alpha = 0.01
    #     # eta = 0.02
    #     # theta = 0.02
    #     # counter = 0
    #     alpha_g = alpha.mean().detach()
    #     eta_g   = eta.mean().detach()
    #     theta_g = theta.mean().detach()
    #     print("alpha", alpha_g)
    #     print("eta_g", eta_g)
    #     print("theta_g", theta_g)
    #     for (name, param), grad in zip(self.named_parameters(), grads):
    #         if self.surprise.get(name, None) is None:
    #             self.surprise[name] = torch.zeros_like(grad)

    #         # print("grad", grad.shape)
    #         # print("self.suprise", self.surprise[name].shape)
    #         self.surprise[name] = self.surprise[name] * eta_g - theta_g * grad
    #         updated_params[name] = alpha_g * param.data + self.surprise[name] if not name[0] in ['K', 'V'] else param.data
    #         param.data = updated_params[name]
    #         # print("counter", counter)
    #         # counter = counter + 1
    #     return loss.item(), updated_params