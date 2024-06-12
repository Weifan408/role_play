import gymnasium as gym
import numpy as np

from ray.rllib.models.torch.misc import same_padding, valid_padding
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "silu":
        return nn.SiLU
    elif activation == "swish":
        return nn.SiLU
    else:
        raise ValueError(f"Activation {activation} is not supported.")


def get_cnn_output_dims(input_dims, cnn_filter_specifiers):
    dims = input_dims  # Creates a copy (works for tuple/list).
    for filter_spec in cnn_filter_specifiers:
        if len(filter_spec) == 3:
            num_filters, kernel, stride = filter_spec
            padding = "same"
        else:
            num_filters, kernel, stride, padding = filter_spec

        if padding == "same":
            _, dims = same_padding(dims[:2], kernel, stride)
        else:
            dims = valid_padding(dims[:2], kernel, stride)

        dims = [dims[0], dims[1], num_filters]
    return (int(np.prod(dims)),)


def get_output_dim_by_action_space(action_space) -> int:
        if isinstance(action_space, gym.spaces.Discrete):
            return action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            return action_space.shape[0]
        else:
            raise NotImplementedError(f"Action space {action_space} not supported yet")



