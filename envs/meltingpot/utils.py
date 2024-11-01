# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MeltingPotEnv as a MultiAgentEnv wrapper to interface with RLLib."""
import functools
from typing import Tuple, Any, Mapping
import tree

import dm_env
import dmlab2d
from gymnasium import spaces
from gymnasium import utils as gym_utils
import matplotlib.pyplot as plt
from meltingpot import substrate
from meltingpot.utils.policies import policy
from ml_collections import config_dict
import numpy as np
from pettingzoo import utils as pettingzoo_utils
from ray.rllib import algorithms
from ray.rllib.env import multi_agent_env
from ray.rllib.policy import sample_batch


PLAYER_STR_FORMAT = "player_{index}"
_WORLD_PREFIX = "WORLD."
_IGNORE_KEYS = [
    "WORLD.RGB",
    "INTERACTION_INVENTORIES",
    # "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
    "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP",
    "PLAYER_ATE_APPLE",
    "PLAYER_CLEANED",
]


class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
    """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

    def __init__(self, env: dmlab2d.Environment):
        """Initializes the instance.

        Args:
        env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
        """
        self._env = env
        self._num_players = len(self._env.observation_spec())
        self._ordered_agent_ids = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self._num_players)
        ]
        # RLLib requires environments to have the following member variables:
        # observation_space, action_space, and _agent_ids
        self._agent_ids = set(self._ordered_agent_ids)
        # RLLib expects a dictionary of agent_id to observation or action,
        # Melting Pot uses a tuple, so we convert
        self.observation_space = self._convert_spaces_tuple_to_dict(
            spec_to_space(self._env.observation_spec()), remove_world_observations=True
        )
        self.action_space = self._convert_spaces_tuple_to_dict(
            spec_to_space(self._env.action_spec())
        )
        super().__init__()

    def reset(self, *args, **kwargs):
        """See base class."""
        timestep = self._env.reset()
        return timestep_to_observations(timestep), {}

    def step(self, action_dict):
        """See base class."""
        actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
        timestep = self._env.step(actions)
        rewards = {
            agent_id: timestep.reward[index]
            for index, agent_id in enumerate(self._ordered_agent_ids)
        }
        done = {"__all__": timestep.last()}
        info = {}

        observations = timestep_to_observations(timestep)
        return observations, rewards, done, done, info

    def close(self):
        """See base class."""
        self._env.close()

    def get_dmlab2d_env(self):
        """Returns the underlying DM Lab2D environment."""
        return self._env

    # Metadata is required by the gym `Env` class that we are extending, to show
    # which modes the `render` method supports.
    metadata = {"render.modes": ["rgb_array"]}

    def render(self) -> np.ndarray:
        """Render the environment.

        This allows you to set `record_env` in your training config, to record
        videos of gameplay.

        Returns:
            np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
            representing RGB values for an x-by-y pixel image, suitable for turning
            into a video.
        """
        observation = self._env.observation()
        world_rgb = observation[0]["WORLD.RGB"]

        # RGB mode is used for recording videos
        return world_rgb

    def _convert_spaces_tuple_to_dict(
        self, input_tuple: spaces.Tuple, remove_world_observations: bool = False
    ) -> spaces.Dict:
        """Returns spaces tuple converted to a dictionary.

        Args:
        input_tuple: tuple to convert.
        remove_world_observations: If True will remove non-player observations.
        """
        return spaces.Dict(
            {
                agent_id: (
                    remove_world_observations_from_space(input_tuple[i])
                    if remove_world_observations
                    else input_tuple[i]
                )
                for i, agent_id in enumerate(self._ordered_agent_ids)
            }
        )


def env_creator(env_config):
    """Outputs an environment for registering."""
    env_config = config_dict.ConfigDict(env_config)
    env = substrate.build(env_config["substrate"], roles=env_config["roles"])
    env = MeltingPotEnv(env)
    return env


class RayModelPolicy(policy.Policy[policy.State]):
    """Policy wrapping an RLLib model for inference.

    Note: Currently only supports a single input, batching is not enabled
    """

    def __init__(
        self,
        model: algorithms.Algorithm,
        policy_id: str = sample_batch.DEFAULT_POLICY_ID,
    ) -> None:
        """Initialize a policy instance.

        Args:
        model: An rllib.trainer.Trainer checkpoint.
        policy_id: Which policy to use (if trained in multi_agent mode)
        """
        self._model = model
        self._prev_action = 0
        self._policy_id = policy_id

    def step(
        self, timestep: dm_env.TimeStep, prev_state: policy.State
    ) -> Tuple[int, policy.State]:
        """See base class."""
        observations = {
            key: value
            for key, value in timestep.observation.items()
            if "WORLD" not in key
        }

        action, state, _ = self._model.compute_single_action(
            observations,
            prev_state,
            policy_id=self._policy_id,
            prev_action=self._prev_action,
            prev_reward=timestep.reward,
        )

        self._prev_action = action
        return action, state

    def initial_state(self) -> policy.State:
        """See base class."""
        self._prev_action = 0
        return self._model.get_policy(self._policy_id).get_initial_state()

    def close(self) -> None:
        """See base class."""


def timestep_to_observations(timestep: dm_env.TimeStep) -> Mapping[str, Any]:
    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value for key, value in observation.items() if key not in _IGNORE_KEYS
        }
    return gym_observations


def remove_unrequired_observations_from_space(observation: spaces.Dict) -> spaces.Dict:
    """Remove observations that are not supposed to be used by policies."""

    return spaces.Dict(
        {key: observation[key] for key in observation if key not in _IGNORE_KEYS}
    )


def spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
    """Converts a dm_env nested structure of specs to a Gym Space.

    BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
    Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

    Args:
      spec: The nested structure of specs

    Returns:
      The Gym space corresponding to the given spec.
    """
    if isinstance(spec, dm_env.specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.floating):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        elif np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        else:
            raise NotImplementedError(f"Unsupported dtype {spec.dtype}")
    elif isinstance(spec, (list, tuple)):
        return spaces.Tuple([spec_to_space(s) for s in spec])
    elif isinstance(spec, dict):
        return spaces.Dict({key: spec_to_space(s) for key, s in spec.items()})
    else:
        raise ValueError("Unexpected spec of type {}: {}".format(type(spec), spec))
