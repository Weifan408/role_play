from typing import Callable, Dict, List, Optional, Union, Tuple

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.utils import get_activation_fn, get_initializer_fn

torch, nn = try_import_torch()


class DenseInptMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layer_dims: List[int],
        hidden_layer_activation: Union[str, Callable] = "relu",
        hidden_layer_use_bias: bool = True,
        hidden_layer_use_layernorm: bool = False,
        output_dim: Optional[int] = None,
        output_use_bias: bool = True,
        output_activation: Union[str, Callable] = "linear",
        dense_input_dim: int = 0,
    ):
        super().__init__()
        assert input_dim > 0

        self.input_dim = input_dim
        self.dense_input_dim = dense_input_dim
        hidden_activation = get_activation_fn(
            hidden_layer_activation, framework="torch"
        )
        self.layers = nn.ModuleList()
        dims = (
            [self.input_dim]
            + list(hidden_layer_dims)
            + ([output_dim] if output_dim else [])
        )
        ipt_dim = self.input_dim
        for i in range(0, len(dims) - 1):
            is_output_layer = output_dim is not None and i == len(dims) - 2
            layers = []
            layer = nn.Linear(
                ipt_dim,
                dims[i + 1],
                bias=output_use_bias if is_output_layer else hidden_layer_use_bias,
            )
            ipt_dim = dims[i + 1] + self.dense_input_dim
            layers.append(layer)

            if not is_output_layer:
                if hidden_layer_use_layernorm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                if hidden_activation is not None:
                    layers.append(hidden_activation())
            self.layers.append(nn.Sequential(*layers))

        output_activation = get_activation_fn(output_activation, framework="torch")
        if output_dim is not None and output_activation is not None:
            self.layers.append(output_activation())

        self.expected_input_dtype = torch.float32

    def forward(self, x):
        # x = x.type(self.expected_input_dtype)
        if self.dense_input_dim > 0:
            for layer in self.layers[:-1]:
                x = layer(x)
                x = torch.cat([x, x[..., -self.dense_input_dim :]], dim=-1)
            x = self.layers[-1](x)
        else:
            for layer in self.layers:
                x = layer(x)
        return x
