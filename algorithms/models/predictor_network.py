from typing import Optional
import gymnasium as gym
from gymnasium.spaces import Box

from ray.rllib.utils.framework import try_import_torch

from ray.rllib.core.models.torch.primitives import TorchMLP

torch, nn = try_import_torch()


class PredicotrNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_classes: int) -> None:
        super().__init__()

        self.mlp = TorchMLP(
            input_dim=input_dim,
            hidden_layer_dims=[],
            output_dim=64,
        )

        self.predictor = TorchMLP(
            input_dim=64,
            hidden_layer_dims=[],
            output_dim=output_dim,
            # output_activation=nn.Linear,
        )
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.mlp(inputs)
        logits = self.predictor(out)
        logits = logits.reshape(*logits.shape[:-1], self.num_classes)
        probs = torch.softmax(logits, dim=-1)
        probs = 0.99 * probs + (1 - 0.99) * 1.0 / self.num_classes

        return probs
