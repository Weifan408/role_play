# Copyright 2022 DeepMind Technologies Limited.
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
"""PettingZoo interface to meltingpot environments."""

import cv2
import functools
import importlib
import numpy as np

from gymnasium import utils as gym_utils
from gymnasium.spaces import Box, Space
import matplotlib.pyplot as plt
from meltingpot import substrate
from ml_collections import config_dict
from pettingzoo import utils as pettingzoo_utils
from pettingzoo.utils import wrappers

from envs.meltingpot.utils import (
    timestep_to_observations,
    remove_world_observations_from_space,
    spec_to_space,
)


PLAYER_STR_FORMAT = "player_{index}"
MAX_CYCLES = 1000


def parallel_env(env_config, max_cycles=MAX_CYCLES):
    return _ParallelEnv(env_config, max_cycles)


def raw_env(env_config, max_cycles=MAX_CYCLES):
    return pettingzoo_utils.parallel_to_aec_wrapper(
        parallel_env(env_config, max_cycles)
    )


def env(env_config, max_cycles=MAX_CYCLES):
    aec_env = raw_env(env_config, max_cycles)
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env


class _MeltingPotPettingZooEnv(pettingzoo_utils.ParallelEnv):
    """An adapter between Melting Pot substrates and PettingZoo's ParallelEnv."""

    def __init__(self, env_config, max_cycles):
        self.env_config = config_dict.ConfigDict(env_config)
        self.max_cycles = max_cycles
        self._env = substrate.build_from_config(
            self.env_config, roles=self.env_config.default_player_roles[:2]
        )
        self._num_players = len(self._env.observation_spec())
        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self._num_players)
        ]
        observation_space = remove_world_observations_from_space(
            spec_to_space(self._env.observation_spec()[0])
        )
        self.observation_space = functools.lru_cache(maxsize=None)(
            lambda agent_id: observation_space
        )
        action_space = spec_to_space(self._env.action_spec()[0])
        self.action_space = functools.lru_cache(maxsize=None)(
            lambda agent_id: action_space
        )
        self.state_space = spec_to_space(self._env.observation_spec()[0]["WORLD.RGB"])
        self.total_reward = {agent: 0 for agent in self.possible_agents}

    def state(self):
        return self._env.observation()

    def reset(self, seed=None):
        """See base class."""
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.total_reward = {agent: 0 for agent in self.possible_agents}
        self.num_cycles = 0
        return timestep_to_observations(timestep), {}

    def step(self, action):
        """See base class."""
        actions = [action[agent] for agent in self.agents]
        timestep = self._env.step(actions)
        rewards = {
            agent: timestep.reward[index] for index, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        done = timestep.last() or self.num_cycles >= self.max_cycles
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if done:
            self.agents = []

        observations = timestep_to_observations(timestep)
        return observations, rewards, dones, dones, infos

    def close(self):
        """See base class."""
        self._env.close()

    def render(self, mode="human", filename=None):
        rgb_arr = self.state()[1]["WORLD.RGB"]
        if mode == "human":
            plt.cla()
            plt.imshow(rgb_arr, interpolation="nearest")
            if filename is None:
                plt.show(block=False)
            else:
                plt.savefig(filename)
            return None
        return rgb_arr


class MeltingPotWrapper(_MeltingPotPettingZooEnv):
    def __init__(self, env_config, max_cycles):

        self.env_config = config_dict.ConfigDict(env_config)
        self.max_cycles = max_cycles
        self.obs_scale = env_config.get("obs_scale", 8)
        self.substrate_name = env_config["name"]
        default_config = self.get_default_config(self.substrate_name)
        with default_config.unlocked():
            default_config.update(env_config)
        self.env_config = default_config
        self._env = substrate.build_from_config(
            self.env_config, roles=self.env_config.default_player_roles
        )
        self.num_players = len(self._env.observation_spec())
        self.possible_agents = [
            PLAYER_STR_FORMAT.format(index=index) for index in range(self.num_players)
        ]
        self.players_ids = self.possible_agents
        self.state_space = spec_to_space(self._env.observation_spec()[0]["WORLD.RGB"])
        self.total_rewards = {agent_id: 0 for agent_id in self.possible_agents}

    @property
    @functools.lru_cache(maxsize=None)
    def action_space(self) -> Space:
        return spec_to_space(self._env.action_spec()[0])

    @property
    @functools.lru_cache(maxsize=None)
    def observation_space(self) -> Space:
        obs_space = remove_world_observations_from_space(
            spec_to_space(self._env.observation_spec()[0])
        )
        obs_space["RGB"] = Box(
            0,
            255,
            (
                obs_space["RGB"].shape[0] // self.obs_scale,
                obs_space["RGB"][1] // self.obs_scale,
                3,
            ),
            dtype=np.uint8,
        )
        obs_space.spaces.pop("RGB", None)
        return obs_space

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed)
        self.total_rewards = {agent_id: 0 for agent_id in self.possible_agents}
        return self.process_obs(obs), info

    def step(self, actions):
        observations, rewards, dones, _, infos = super().step(actions)
        dones["__all__"] = np.any(list(dones.values()))
        self.total_rewards = {
            agent_id: self.total_rewards[agent_id] + rewards[agent_id]
            for agent_id in self.possible_agents
        }
        return self.process_obs(observations), rewards, dones, dones, infos

    def process_obs(self, obs):
        for player_id in obs:
            obs[player_id]["RGB"] = cv2.resize(
                obs[player_id]["RGB"],
                (
                    obs[player_id]["RGB"].shape[0] // self.obs_scale,
                    obs[player_id]["RGB"].shape[0] // self.obs_scale,
                ),
                interpolation=cv2.INTER_AREA,
            )
        return obs

    def get_default_config(self, substrate: str):
        path = f"envs.meltingpot.configs.{substrate}"
        module = importlib.import_module(path)
        config = module.get_config()
        with config.unlocked():
            config.lab2d_settings_builder = module.build
        return config.lock()


class _ParallelEnv(MeltingPotWrapper, gym_utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_config, max_cycles):
        gym_utils.EzPickle.__init__(self, env_config, max_cycles)
        MeltingPotWrapper.__init__(self, env_config, max_cycles)
