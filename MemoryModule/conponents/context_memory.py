from torch import nn
import torch
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from MemoryModule.conponents.context_memory_layer import ContextMemoryLayer
# import torch.functional as F
class ContextMemory(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        context_window,
        pm_len,
        n_layers = 1,
        n_layers_nmm = 2,

    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_window = context_window

        self.emb_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            ContextMemoryLayer(
                hidden_dim,
                context_window,
                pm_len,
                n_layers_nmm=n_layers_nmm
            )
            for _ in range(n_layers)
        ])

        self.final_layer = nn.Linear(hidden_dim, output_dim)

        self.silu = nn.SiLU()

        self.outer_params = list(self.emb_layer.parameters()) + list(self.final_layer.parameters())
        for layer in self.layers:
            self.outer_params += layer.outer_params

    # Simple taking an input of shape (batch_size, context_len, input_dim)
    # and returns (batch_size, output_dim)
    def process(self, x, alpha, theta, eta):
        # print("hidden state insie", x)
        batch_size = x.shape[0]

        # print("shape of x in process:", x.shape)

        # Pass x through the embedding layer to get (batch_size, context_len, hidden_dim)
        # print("shape of x before emb_layer:", x.reshape(-1, self.input_dim).shape)
        # print("input dim:", self.input_dim)
        x = self.emb_layer(x.reshape(-1, self.input_dim)).view(batch_size, -1, self.hidden_dim)
        # print("shape after emb_layer:", x.shape)

        # Pass x thorugh all the MACTitanLayer's
        for layer in self.layers:
            x = x + self.silu(layer(x, alpha, theta, eta))

        # print("shape of x after all layers:", x)

        return self.final_layer(x)


    def forward(self, x, alpha, theta, eta):
        # We are given a batch of long sequences (batch_size, N, input_dim)
        # The output should be of the size (batch_size, N, output_dim)
        # res = torch.zeros((x.shape[0], x.shape[1], self.output_dim)).cuda()

        # Note that in MAC, to evaluate y_{t}, we must have processed chunks
        # x_{t-context_window:t}, x_{t-2*context_window:t-context_window} etc.

        # Let's then consider this problem as context_window subproblems, based
        # on the position index modulo context_window

        # x = np.permute_dims(sliding_window_view(x.cpu(), self.context_window, axis=1), (0,1,3,2))
        # print("shape of x:", x.shape)
        # x = x.unfold(dimension=1, size=self.context_window, step=1)
        # x = x.permute(0, 1, 3, 2)
        # print("shape of x after sliding window:", x.shape)
        # residual = x.shape[1] % self.context_window

        # stz = torch.from_numpy(x[:,:-residual].reshape(x.shape[0], -1, self.context_window, self.context_window, self.input_dim)).cuda()
        # stz = x[:, :-residual].reshape(
        #         x.shape[0], -1, self.context_window, self.context_window, self.input_dim
        #     ).cuda()
        # # stz: (batch_size, N//context_window, context_window, context_window, input_dim)
        # print("shape of stz:", stz.shape)

        # for i in range(stz.shape[1]):
        #     slide = stz[:,i].reshape(-1, self.context_window, x.shape[-1])
        #     # slide: (batch_size * context_window, context_window, input_dim)
        #     out = self.process(slide).reshape(-1, self.context_window, self.output_dim)
        #     res[:, (i+1)*self.context_window -1:(i+2)*self.context_window -1] = out
        
        # # print("shape of res after processing full windows:", res.shape)
        # # residual_part: (batch_size, context_window - 1 + N % context_window, context_window, input_dim)
        # residual_part = x[:,-residual:].reshape(-1, self.context_window, self.input_dim).cuda()
        # print("shape of residual_part:", x.shape)
        res_out = self.process(x, alpha, theta, eta).reshape(-1, x.shape[1], self.output_dim)
        # res[:, -residual:] = res_out

        return res_out