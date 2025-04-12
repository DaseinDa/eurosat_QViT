
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit.visualization import *
from qiskit_aer import AerSimulator

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer.primitives import Sampler

from qiskit_algorithms.utils import algorithm_globals

import utils as U
import circuits as C

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class quantumAttentionBlock(nn.Module):

    def __init__(self, vec_loader, matrix_mul, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

        aersim = AerSimulator(method='statevector', device='GPU')
        sampler = Sampler()
        sampler.set_options(backend=aersim)

        self.vx = C.Vx(embed_dim, vec_loader, matrix_mul)
        qc, num_weights = self.vx()
        self._vx = TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:], weight_params=qc.parameters[:num_weights], input_gradients=True, sampler = sampler))

        qc, num_weights = C.xWx(embed_dim, vec_loader, matrix_mul)()
        self._xwx = TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:]+qc.parameters[:embed_dim-1], weight_params=qc.parameters[embed_dim -1:-embed_dim+1], input_gradients=True, sampler = sampler))

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, D = x.shape

        inp_x = self.layer_norm_1(x)

        vx = torch.empty((B, D, N)).to(device)
        xwx = torch.empty((B, N, N)).to(device)

        inp_x = inp_x/torch.sqrt(torch.sum(torch.pow(inp_x, 2)+1e-4, dim=1, keepdim=True)+1e-8)
        parameters = self.vx.get_RBS_parameters(inp_x)

        ei = [2**j for j in range(0, D)]
        for i in range(N):
            vx[:, :, i] = self._vx(parameters[:, i, :])[:, ei]

        ei = [j for j in range(1, 2**D, 2)]
        for i in range(N):
            for j in range(N):
                p = torch.cat((parameters[:, i, :], parameters[:, j, :]), dim=1)
                xwx[:, i, j] = torch.sum(self._xwx(p)[:, ei], dim=1)

        vx = torch.sqrt(vx + 1e-8)
        xwx = torch.sqrt(xwx + 1e-8)

        attn = F.softmax(xwx, dim=-1)
        t = torch.matmul(attn, vx.transpose(1,2))
        x = x + t
        x = x + self.linear(self.layer_norm_2(x))

        return x


class QuantumVisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, dropout=0.0, vec_loader='diagonal', matrix_mul='pyramid'):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)
        self.transformer = nn.Sequential(*[
            quantumAttentionBlock(vec_loader, matrix_mul, embed_dim, hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = None  # <-- 延迟初始化

    def forward(self, x):
        x = U.img_to_patch(x, self.patch_size)
        B, T, _ = x.shape  # T = 实际 patch 数

        x = self.input_layer(x)

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        # 动态初始化 positional embedding
        if self.pos_embedding is None or self.pos_embedding.shape[1] != T + 1:
            self.pos_embedding = nn.Parameter(torch.randn(1, T + 1, self.embed_dim).to(x.device))

        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)

        cls = x[:, 0, :]
        out = self.mlp_head(cls)
        return out
