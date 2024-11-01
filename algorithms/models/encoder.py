from typing import Tuple, Optional

import gymnasium as gym
from gymnasium.spaces import Box, Dict

from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.models.torch.primitives import TorchMLP
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict

from algorithms.models.dict_obs_encoder import DictObsEncoder
from algorithms.models.gru import GRU

torch, nn = try_import_torch()


class Encoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ) -> None:
        super().__init__()

        self.use_rnn = model_config_dict["use_lstm"]
        if isinstance(observation_space, Dict):
            self.encoder = DictObsEncoder(
                observation_space=observation_space,
                model_config_dict=model_config_dict,
            )
            self.encoder_latent_dim = self.encoder.output_dim
        elif isinstance(observation_space, Box) and len(observation_space.shape) == 3:
            raise NotImplementedError("CNN not supported yet")
        elif isinstance(observation_space, Box) and len(observation_space.shape) == 1:
            if model_config_dict["encoder_latent_dim"]:
                hidden_layer_dims = model_config_dict["fcnet_hiddens"]
                self.encoder_latent_dim = model_config_dict["encoder_latent_dim"]
            else:
                hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
                self.encoder_latent_dim = model_config_dict["fcnet_hiddens"][-1]

            self.encoder = TorchMLP(
                input_dim=observation_space.shape[0],
                hidden_layer_dims=hidden_layer_dims,
                hidden_layer_use_layernorm=True,
                output_dim=self.encoder_latent_dim,
                hidden_layer_activation=model_config_dict["fcnet_activation"],
            )

        if self.use_rnn:
            self.gru = GRU(
                input_dim=self.encoder_latent_dim,
                hidden_size=model_config_dict["lstm_cell_size"],
                num_layers=model_config_dict["custom_model_config"].get(
                    "num_lstm_layers", 1
                ),
            )
            self.encoder_latent_dim = model_config_dict["lstm_cell_size"]

    def forward(
        self, inputs: NestedDict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        encoder_out = self.encoder(inputs)

        if self.use_rnn:
            encoder_out, h_next = self.gru(
                encoder_out, inputs[Columns.STATE_IN][ACTOR]["h"]
            )
        else:
            h_next = None

        return encoder_out, h_next

    def get_initial_state(self) -> Optional[dict]:
        if self.use_rnn:
            return self.gru.get_initial_state()

        return None


class ACEncoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ) -> None:
        super().__init__()

        self.shared = model_config_dict["vf_share_layers"]
        if self.shared:
            self.actor_encoder = self.critic_encoder = Encoder(
                observation_space=observation_space,
                action_space=action_space,
                model_config_dict=model_config_dict,
            )
        else:
            self.actor_encoder = Encoder(
                observation_space=observation_space,
                action_space=action_space,
                model_config_dict=model_config_dict,
            )

            self.critic_encoder = Encoder(
                observation_space=observation_space,
                action_space=action_space,
                model_config_dict=model_config_dict,
            )

        self.actor_encoder_latent_dim = self.actor_encoder.encoder_latent_dim
        self.vf_encoder_latent_dim = self.critic_encoder.encoder_latent_dim

    def forward(self, inputs):
        actor_encoder_out, actor_h_next = self.actor_encoder(inputs)
        critic_encoder_out, critic_h_next = self.critic_encoder(inputs)

        output = {
            ENCODER_OUT: {
                ACTOR: actor_encoder_out,
                CRITIC: critic_encoder_out,
            }
        }
        if actor_h_next is not None:
            output[Columns.STATE_OUT] = {
                ACTOR: {"h": actor_h_next},
                CRITIC: {"h": critic_h_next},
            }
        return output

    def get_initial_state(self) -> dict:
        initial_state = {}
        if self.actor_encoder.use_rnn:
            initial_state[ACTOR] = self.actor_encoder.get_initial_state()
        if self.critic_encoder.use_rnn:
            initial_state[CRITIC] = self.critic_encoder.get_initial_state()
        return initial_state
