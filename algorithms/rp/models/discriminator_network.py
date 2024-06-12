from typing import Optional
import gymnasium as gym
from gymnasium.spaces import Box

from ray.rllib.utils.framework import try_import_torch

from ray.rllib.core.models.torch.primitives import TorchMLP

torch, nn = try_import_torch()


class DiscriminatorNetwork(nn.Module):
    def __init__(self,
            input_dim: int,
            output_dim: int,
        ) -> None:
        super().__init__()

        self.lower_bound = -torch.pi
        self.upper_bound = torch.pi
        self.num_buckets = output_dim
        self.net = TorchMLP(
            input_dim=input_dim,
            hidden_layer_dims=[],
            output_dim=output_dim,
            # output_activation=nn.Linear,
        )

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        out_put = self.net(inputs)

        return out_put

