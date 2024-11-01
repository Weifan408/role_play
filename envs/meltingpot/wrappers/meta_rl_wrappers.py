from collections import defaultdict
from typing import Any, Dict, List, SupportsFloat, TypeVar
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from envs.utils import obs_transform
from utils import two_hot

ObsType = TypeVar("ObsType")


class meta_rl_env(MultiAgentEnv):
    def __init__(self, flat_env, env_config):
        self.flat_env = flat_env

        self.role_class_num = 8
        self.num_players = self.flat_env._num_players
        self.players_ids = self.flat_env._agent_ids

        self.trail_length = env_config.get("trial_length", 10)
        # self.max_cycles = env_config.get("max_cycles", 1000)
        self.rgb_obs = env_config.get("rgb_obs", False)
        self.eval = env_config.get("eval", False)
        self.w = env_config.get("w", 0.3)

        self.episode_total_rewards = {player_id: 0 for player_id in self.players_ids}
        self.reward_fn = self.reward_fn

        self.step_num = 0
        self.episode_cnt = 0
        self.trial_infos = {
            player_id: defaultdict(list) for player_id in self.players_ids
        }

        self.phase = 0
        self.theta_map = np.linspace(-np.pi, np.pi, self.role_class_num, endpoint=False)
        self.theta_idx = {
            player_id: np.argmin(np.abs(self.theta_map - np.pi / 4))
            for player_id in self.players_ids
        }
        self.svo_phase = {
            0: np.argmin(np.abs(self.theta_map - np.pi / 4)),
            1: [i for i in range(self.role_class_num)],
        }

    # r = w * r + (1-w) * (r * cos(theta) + (R-r) * sin(theta))
    def reward_fn(self) -> Dict[str, float]:
        cur_total_reward = sum(self.original_rewards.values())

        new_rewards = {}
        for player_id, r_i in self.original_rewards.items():
            avg_cur_r_minus_i = (cur_total_reward - r_i) / (self.num_players - 1)
            theta = self.theta_map[self.theta_idx[player_id]]
            new_rewards[player_id] = np.round(
                self.w * self.original_rewards[player_id]
                + (1 - self.w)
                * (r_i * np.cos(theta) + (avg_cur_r_minus_i * np.sin(theta))),
                decimals=2,
            )

        return new_rewards

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        self.trial_infos = {
            player_id: defaultdict(list) for player_id in self.players_ids
        }

        obs, info = self.flat_env.reset()
        if not self.eval:
            self.theta_idx = {
                "player_0": np.random.choice(self.svo_phase[self.phase]),
                "player_1": np.random.choice(self.svo_phase[self.phase]),
            }
        self.original_rewards = None
        self.init_compute_total_rewards()
        obs = obs_transform(obs, self.theta_idx, 0)
        self.episode_cnt = 0
        return obs, info

    def step(
        self, action_dict: Dict
    ) -> tuple[ObsType, SupportsFloat, bool, dict[str, Any]]:

        obs, original_rewards, terminated, truncated, infos = self.flat_env.step(
            action_dict
        )

        self.original_rewards = original_rewards
        rewards = self.reward_fn()
        for player_id in self.players_ids:
            self.compute_total_rewards[player_id] += rewards[player_id]

        dones = {"__all__": False}
        if terminated["__all__"] or truncated["__all__"]:
            for player_id in self.players_ids:
                if player_id not in obs:
                    obs[player_id] = {}

            for player_id in self.players_ids:
                infos[player_id]["agent_total_reward"] = np.round(
                    self.flat_env.total_rewards[player_id], decimals=2
                )
                infos[player_id]["compute_total_reward"] = np.round(
                    self.compute_total_rewards[player_id], decimals=2
                )

                self.trial_infos[player_id]["compute_total_reward"].append(
                    infos[player_id]["compute_total_reward"]
                )
                self.trial_infos[player_id]["agent_total_reward"].append(
                    infos[player_id]["agent_total_reward"]
                )

            infos["__common__"] = True
            self.init_compute_total_rewards()
            self.episode_cnt += 1
            if self.episode_cnt < self.trail_length:
                obs, _ = self.flat_env.reset()
                obs = obs_transform(obs, self.theta_idx, 1)
            else:
                dones["__all__"] = True
                obs = obs_transform(obs, self.theta_idx, 0)
                infos = self.trial_infos
        else:
            obs = obs_transform(obs, self.theta_idx, 0)

        return obs, rewards, dones, dones, infos

    def set_theta_idx(self, theta_idx: Dict[str, float]):
        self.theta_idx = theta_idx

    def print_phase(self):
        print(self.phase)

    def set_phase(self, phase):
        self.phase = phase

    def render(self):
        return self.flat_env.render()

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                **self.flat_env.single_agent_observation_space,
                "theta": gym.spaces.Discrete(self.role_class_num),
                "other_theta": gym.spaces.Tuple(
                    [gym.spaces.Discrete(8) for _ in range(self.num_players - 1)]
                ),
                "done": gym.spaces.Discrete(2),
            }
        )

    @property
    def action_space(self):
        return self.flat_env.single_agent_action_space

    def init_compute_total_rewards(self):
        self.compute_total_rewards = {player_id: 0 for player_id in self.players_ids}


class no_meta_rl_env(meta_rl_env):
    def __init__(self, flat_env, env_config):
        super().__init__(flat_env, env_config)

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.flat_env.reset()
        if not self.eval:
            self.theta_idx = {
                player_id: np.random.choice(self.svo_phase[self.phase])
                for player_id in self.players_ids
            }

        self.original_rewards = None
        self.init_compute_total_rewards()
        obs = obs_transform(obs, self.theta_idx, 0)
        self.episode_cnt = 0
        self.step_num = 0
        return obs, info

    def step(
        self, action_dict: Dict
    ) -> tuple[ObsType, SupportsFloat, bool, dict[str, Any]]:
        obs, original_rewards, terminated, truncated, infos = self.flat_env.step(
            action_dict
        )

        self.original_rewards = original_rewards
        rewards = self.reward_fn()
        for player_id in self.players_ids:
            self.compute_total_rewards[player_id] += rewards[player_id]
        self.step_num += 1

        dones = {"__all__": False}
        if terminated["__all__"] or truncated["__all__"]:
            for player_id in self.players_ids:
                infos[player_id]["agent_total_reward"] = np.round(
                    self.flat_env.total_rewards[player_id], decimals=2
                )
                infos[player_id]["compute_total_reward"] = np.round(
                    self.compute_total_rewards[player_id], decimals=2
                )

            self.init_compute_total_rewards()
            dones["__all__"] = True
            obs = obs_transform(obs, self.theta_idx, 0)
        else:
            obs = obs_transform(obs, self.theta_idx, 0)

        return obs, rewards, dones, dones, infos


class ObsRewriteEnv(gym.core.ObservationWrapper, MultiAgentEnv):
    def __init__(self, env: Dict, env_config: Dict) -> None:
        super().__init__(env)
        self.preprocessors = dict()
        self._combine_keys = env_config.get("combine_obs_keys", [])
        self._other_keys = env_config.get("other_obs_keys", [])

        self._observation_space = self.rewrite_observation_space()

    def rewrite_observation_space(self) -> gym.Space:
        original_spaces = self.env.observation_space
        spaces = self.preprocess_spaces(original_spaces)
        return spaces

    def observation(self, observation: Dict):
        new_obs = {}
        for id, obs in observation.items():
            single_player_new_obs = {}

            _combine_values = {
                key: value for key, value in obs.items() if key in self._combine_keys
            }
            for key, value in obs.items():
                if key in self._other_keys:
                    single_player_new_obs[key] = self.preprocessors[key].transform(
                        value
                    )

                if key in self._combine_keys:
                    _combine_values[key] = value

            single_player_new_obs["mlp"] = self.preprocessors["mlp"].transform(
                _combine_values
            )

            new_obs[id] = single_player_new_obs
        return new_obs

    def preprocess_spaces(self, spaces):
        other_spaces = gym.spaces.Dict()
        mlp_spaces = gym.spaces.Dict()

        for key, space in spaces.items():
            if key in self._combine_keys:
                mlp_spaces[key] = space
            if key in self._other_keys:
                preprocessor = ModelCatalog.get_preprocessor_for_space(space)
                other_spaces[key] = preprocessor.observation_space
                self.preprocessors[key] = preprocessor

        mlp_preprocessor = ModelCatalog.get_preprocessor_for_space(mlp_spaces)
        self.preprocessors["mlp"] = mlp_preprocessor

        return gym.spaces.Dict(
            {
                "mlp": mlp_preprocessor.observation_space,
                **other_spaces,
            }
        )

    def process_obs(self, observation):
        new_obs = {}
        _combine_values = {
            key: value
            for key, value in observation.items()
            if key in self._combine_keys
        }
        for key, value in observation.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 3:
                value = value / 255  # Normalize the image

            if key in self._other_keys:
                new_obs[key] = self.preprocessors[key].transform(value)

            if key in self._combine_keys:
                _combine_values[key] = value

        new_obs["mlp"] = self.preprocessors["mlp"].transform(_combine_values)
        return new_obs
