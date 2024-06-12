import functools

import gymnasium as gym
from gymnasium.spaces import Box
from typing import TYPE_CHECKING

from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.torch.primitives import TorchMLP


if TYPE_CHECKING:
    from ray.rllib.core.models.base import Model, Encoder

from algorithms.rp.models.encoder import ACEncoder
from algorithms.rp.models.discriminator_network import DiscriminatorNetwork
from algorithms.rp.utils import get_output_dim_by_action_space


class RPCatalog(Catalog):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            model_config_dict: dict,
    ):
        self.multiplayers = model_config_dict['custom_model_config'].get("multiplayers", False)
        self.encoder = ACEncoder(observation_space, action_space, model_config_dict)

        two_hot_latent_dim = observation_space['theta'].shape[0] + observation_space['other_theta'].shape[0]
        vf_head_input_dim = self.encoder.vf_encoder_latent_dim + two_hot_latent_dim

        if self.multiplayers:
            other_theta_encoder_dim = 64
            self.other_theta_encoder = TorchMLP(
                input_dim=observation_space['other_theta'].shape[0],
                hidden_layer_dims=[128, 128],
                hidden_layer_use_layernorm= True,
                output_dim=other_theta_encoder_dim,
                hidden_layer_activation="silu",
                # output_activation=,
            )

            self.other_theta_decoder = TorchMLP(
                input_dim=other_theta_encoder_dim,
                hidden_layer_dims=[128, 128],
                hidden_layer_use_layernorm= True,
                output_dim=observation_space['other_theta'].shape[0],
                hidden_layer_activation="SiLU",
                # output_activation=""
            )
        else:
            other_theta_encoder_dim = observation_space['other_theta'].shape[0]

        # pi_head_input_dim = self.encoder.actor_encoder_latent_dim + observation_space['theta'].shape[0]
        pi_head_input_dim = self.encoder.actor_encoder_latent_dim + observation_space['theta'].shape[0] + other_theta_encoder_dim
        self.discriminator = DiscriminatorNetwork(
            input_dim = model_config_dict["lstm_cell_size"] + observation_space['theta'].shape[0],
            output_dim=other_theta_encoder_dim,
        )

        self.pi_head = TorchMLP(
            input_dim=pi_head_input_dim,
            hidden_layer_dims=model_config_dict["post_fcnet_hiddens"],
            hidden_layer_use_layernorm= True,
            output_dim=get_output_dim_by_action_space(action_space),
            hidden_layer_activation=model_config_dict["post_fcnet_activation"],
            # output_activation=,
        )

        self.vf_head = TorchMLP(
            input_dim=vf_head_input_dim,
            hidden_layer_dims=model_config_dict["post_fcnet_hiddens"],
            hidden_layer_use_layernorm= True,
            output_dim=1,
            hidden_layer_activation=model_config_dict["post_fcnet_activation"],
            # output_activation=,
        )
        
        # self.action_distribution_cls = self.get_action_distribution_cls(action_space)
        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space, action_space=action_space
        )