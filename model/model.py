import torch
import torch.nn as nn
from typing import List, Union


class MPGModel(nn.Module):
    def __init__(
            self,
            layer_sizes: List[int] | None = None,
            activation: Union[str, List[str]] = "silu",
            use_batch_norm: bool = False,
            dropout: float = 0.0,
            use_residual: bool = False
    ):
        super(MPGModel, self).__init__()

        if layer_sizes is None:
            layer_sizes = [8, 64, 128, 64, 32, 1]

        self.layer_sizes = layer_sizes
        self.use_residual = use_residual
        self.num_hidden = len(layer_sizes) - 2

        if isinstance(activation, str):
            activation_names = [activation] * self.num_hidden
        else:
            activation_names = list(activation)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if i < self.num_hidden:
                if use_batch_norm:
                    self.norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                else:
                    self.norms.append(nn.Identity())

                self.acts.append(self._build_activation(activation_names[i]))

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        if name == "sigmoid":
            return nn.Sigmoid()
        elif name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif name == "elu":
            return nn.ELU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu":
            return nn.SiLU()
        elif name == "mish":
            return nn.Mish()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "prelu":
            return nn.PReLU()
        else:
            return nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers) - 1):
            identity = x
            x = self.layers[i](x)
            x = self.norms[i](x)
            x = self.acts[i](x)
            x = self.drop(x)

            if self.use_residual and identity.shape[-1] == x.shape[-1]:
                x = x + identity

        x = self.layers[-1](x)

        return x

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_hidden_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        results = []

        for i in range(len(self.layers) - 1):
            identity = x
            x = self.layers[i](x)
            x = self.norms[i](x)
            x = self.acts[i](x)

            if self.use_residual and identity.shape[-1] == x.shape[-1]:
                x = x + identity

            results.append(x.detach().clone())

        return results
