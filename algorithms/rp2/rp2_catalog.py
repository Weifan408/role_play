import functools

import gymnasium as gym
from gymnasium.spaces import Box
from typing import TYPE_CHECKING

from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.torch.primitives import TorchMLP

from algorithms.models.encoder import Encoder
from algorithms.models.mlp import DenseInptMLP
from algorithms.utils import get_output_dim_by_action_space

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class RPCatalog(Catalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        self.actor_encoder = Encoder(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
            cnn_key=model_config_dict["custom_model_config"]["actor_cnn_key"],
            use_rnn=True,
        )
        self.critic_encoder = Encoder(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
            cnn_key=model_config_dict["custom_model_config"]["critic_cnn_key"],
        )
        self.class_num = model_config_dict["custom_model_config"]["class_num"]

        self.predictors = nn.ModuleList()
        self.theta_classes = observation_space["theta"].shape[0]
        for _ in range(self.theta_classes):
            self.predictors.append(
                TorchMLP(
                    input_dim=self.actor_encoder.encoder_latent_dim,
                    hidden_layer_dims=[64],
                    output_dim=self.class_num,
                    # output_activation="Softmax",
                )
            )

        pi_head_input_dim = (
            self.actor_encoder.encoder_latent_dim
            + self.theta_classes * self.class_num * 2
        )
        vf_head_input_dim = (
            self.critic_encoder.encoder_latent_dim
            + self.theta_classes * self.class_num * 2
        )

        hidden_layer_dims = [dim for dim in model_config_dict["post_fcnet_hiddens"]]
        self.pi_head = DenseInptMLP(
            input_dim=pi_head_input_dim,
            hidden_layer_dims=hidden_layer_dims,
            hidden_layer_use_layernorm=True,
            output_dim=get_output_dim_by_action_space(action_space),
            hidden_layer_activation=model_config_dict["post_fcnet_activation"],
            dense_input_dim=self.theta_classes * 2,
            # output_activation=,
        )

        self.vf_head = DenseInptMLP(
            input_dim=vf_head_input_dim,
            hidden_layer_dims=hidden_layer_dims,
            hidden_layer_use_layernorm=True,
            output_dim=1,
            hidden_layer_activation=model_config_dict["post_fcnet_activation"],
            dense_input_dim=self.theta_classes * 2,
            # output_activation=,
        )

        # self.action_distribution_cls = self.get_action_distribution_cls(action_space)
        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space, action_space=action_space
        )
