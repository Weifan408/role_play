from typing import Optional

from ray.rllib.core.models.torch.primitives import TorchMLP
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class GRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        mlp_encoder_out_dim: int = 512,
        mlp_hidden_layer_dims: Optional[list[int]] = None,
    ):
        # in dreamer, they use a pre_mlp to process the input before the gru
        super().__init__()

        self.pre_gru_layer = TorchMLP(
            input_dim=input_dim,
            hidden_layer_use_layernorm=True,
            hidden_layer_dims=mlp_hidden_layer_dims or [],
            hidden_layer_activation=nn.SiLU,
            output_dim=mlp_encoder_out_dim,
        )

        self.gru = nn.GRU(
            input_size=mlp_encoder_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        out = self.pre_gru_layer(inputs)
        h = h.transpose(0, 1)
        out, h_next = self.gru(out, h)
        h_next = h_next.transpose(0, 1)
        return out, h_next

    def get_initial_state(self):
        return {
            'h': torch.zeros(self.gru.num_layers, self.gru.hidden_size)
        }
