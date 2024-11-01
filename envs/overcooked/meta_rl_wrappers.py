from collections import defaultdict
from typing import Any, Dict, List, SupportsFloat, TypeVar

# import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import MultiAgentEnv


# 0,0,0,0,r[-5:-5:1],r[0:0:1],0,0,0,r[0:0:1],0,r[5:5:1],r[0:0:1],0,0,0,0,0,0,r[5:5:1],0,0,r[0:0:1],0,r[0:30:2],r[1:1:1],
# 0,1,0,0,0,0,0,0,0,1
# "0,0,0,0,r[-5:0:2],r[-5:5:3],0,0,0,r[0:5:2],0,r[-5:5:3],r[-5:5:3],0,0,0,0,0,0,r[0:5:2],0,0,r[-5:5:3],0,r[-30:30:3],r[0:1:2]"

# 0,0,0,0,0,r[5:5:1],0,r[5:5:1],0,r[5:5:1],0,0,r[-5:5:3],0,r[-10:10:3],r[-10:0:2],r[0:10:2],0,0,0,0,0,r[-3:3:3],r[-3:3:3],r[-10:0:2],1
# 0,0,0,0,0,r[-5:5:3],0,r[-5:5:3],0,r[-5:5:3],0,0,r[-5:5:3],0,r[-10:10:3],r[-10:0:2],r[0:10:2],0,r[-3:3:3],r[-3:3:3],r[-10:0:2],r[0:1:2]

# w0: "0,0,0,0,r[-5:-5:1],r[-3:3:3],0,0,0,r[0:0:1],0,r[5:5:1],r[0:0:1],0,0,0,0,0,0,r[5:5:1],0,0,r[0:0:1],0,r[0:30:2],r[1:1:1]"
ObsType = TypeVar("ObsType")
SHAPED_INFO_KEYS = [
    "put_onion_on_X",
    "put_tomato_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",  # r[-5:-5:3] adv coor
    "pickup_onion_from_O",  # r[-5:5:3]  adv  coor   r[-5:5:3]
    "pickup_tomato_from_X",
    "pickup_tomato_from_T",  # new  r[-5:5:3]       r[-5:5:3]
    "pickup_dish_from_X",
    "pickup_dish_from_D",  # r[0:5:2]  adv coor     r[-5:5:3]
    "pickup_soup_from_X",
    "USEFUL_DISH_PICKUP",  #
    "SOUP_PICKUP",  #  r[-5:5:3]  # adv ccor counter r[-5:5:3]
    "PLACEMENT_IN_POT",  # r[-5:5:3]
    "viable_placement",  # new                      r[-10:10:3]
    "optimal_placement",  # new                     r[-10:0:2]
    "catastrophic_placement",  # new                r[0:10:2]
    "useless_placement",
    "useful_onion_pickup",
    "useful_onion_drop",  # r[0:5:2]
    "useful_tomato_drop",
    "useful_dish_drop",
    "potting_onion",  # r[-5:5:3] 1                  r[-3:3:3]
    "potting_tomato",  # new                         r[-3:3:3]
    "delivery",  # r[-30:30:3]                       r[-10:0:2]
    "delivery_mix",
    "delivery_onion",
    "delivery_tomato",
    "first_tomato",
    "follow_tomato",
    "useless_tomato_pickup",
]

OLD_SHAPED_INFO_KEYS = [
    "put_onion_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",
    "pickup_onion_from_O",  #
    "pickup_dish_from_X",
    "pickup_dish_from_D",  #
    "pickup_soup_from_X",
    "USEFUL_DISH_PICKUP",
    "SOUP_PICKUP",  #
    "PLACEMENT_IN_POT",  #
    "delivery",  #
    "stay",
]


class meta_rl_env(MultiAgentEnv):
    def __init__(self, flat_env, env_config):
        self.flat_env = flat_env

        self.preference_num = env_config.get("preference_num", 11)
        self.num_players = self.flat_env.num_agents
        self.players_ids = self.flat_env._agent_ids

        self.trail_length = env_config.get("trial_length", 10)
        self.eval = env_config.get("eval", False)

        self.w0_str = env_config.get("w0", None)
        self.w1_str = env_config.get("w1", None)
        self.w0 = None
        self.w0_theta = None
        self.w1 = None
        self.w1_theta = None

        if self.eval:
            assert env_config.get("w0_theta", None) is not None
            self.w0_theta = list(map(int, env_config["w0_theta"].split(",")))
            self.w0 = list(map(int, self.w0_str.split(",")))
            self.w1 = list(map(int, self.w1_str.split(",")))
            self.w1_theta = list(map(int, env_config["w1_theta"].split(",")))

        self.thetas = {
            self.players_ids[0]: self.w0_theta,
            self.players_ids[1]: self.w1_theta,
        }

        self.trail_infos = {
            self.players_ids[i]: defaultdict(list) for i in range(self.num_players)
        }

        self.episode_cnt = 0
        self.step_cnt = 0

    def random_choice_w(self, str):
        thetas = []

        def parse_value(s):
            if s.startswith("r"):
                if "[" in s:
                    s = s[2:-1]
                    l, r, n = s.split(":")
                    l, r, n = float(l), float(r), int(n)
                    s = np.random.choice(np.linspace(l, r, n))
                else:
                    v = float(s[1:])
                    s = np.random.randint(-v, v + 1)
                thetas.append(np.sign(s))
            return int(s)

        w = []
        w_lst = str.split(",")
        for s in w_lst:
            tmp = parse_value(s)
            w.append(tmp)
        return w, np.array(thetas, dtype=np.int64)

    def reset(self, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        self.step_cnt = 0
        self.trail_infos = {
            self.players_ids[i]: defaultdict(list) for i in range(self.num_players)
        }

        obs, info = self.flat_env.reset()
        if not self.eval:
            self.w0, self.w0_theta = self.random_choice_w(self.w0_str)
            self.w1, self.w1_theta = self.random_choice_w(self.w1_str)

        self.thetas = {
            self.players_ids[0]: np.array(self.w0_theta, dtype=np.int64) + 1,
            self.players_ids[1]: np.array(self.w1_theta, dtype=np.int64) + 1,
        }
        self.flat_env.set_w(self.w0, self.w1)

        obs = self.add_additional_obs(obs, False)
        self.episode_cnt = 0
        return obs, info

    def step(
        self, action_dict: Dict
    ) -> tuple[ObsType, SupportsFloat, bool, dict[str, Any]]:
        self.step_cnt += 1
        # action_list = [action_dict[player_id] for player_id in self.players_ids]
        action_list = [
            action_dict[player_id] if player_id in action_dict else None
            for player_id in self.players_ids
        ]
        obs, rewards, terminated, truncated, infos = self.flat_env.step(action_list)
        obs = self.add_additional_obs(obs, terminated["__all__"])

        dones = {"__all__": False}
        if terminated["__all__"] or truncated["__all__"]:
            for player_id in self.players_ids:
                if player_id not in obs:
                    obs[player_id] = {}

            self.episode_cnt += 1
            self.update_trail_infos(infos)
            if self.episode_cnt < self.trail_length:
                tmp = infos
                obs, infos = self.flat_env.reset()
                infos["episode"] = tmp["agent_0"]["episode"]
                obs = self.add_additional_obs(obs, True)
            else:
                dones["__all__"] = True
                for player_id in self.players_ids:
                    infos[player_id]["trial"] = self.trail_infos[player_id]
        return obs, rewards, dones, dones, infos

    def render(self, render_mode="rgb_array"):
        return self.flat_env.render(render_mode)

    def add_additional_obs(self, obs, done):
        obs[self.players_ids[0]]["theta"] = self.thetas[self.players_ids[0]]
        obs[self.players_ids[0]]["other_theta"] = self.thetas[self.players_ids[1]]
        obs[self.players_ids[1]]["theta"] = self.thetas[self.players_ids[1]]
        obs[self.players_ids[1]]["other_theta"] = self.thetas[self.players_ids[0]]

        obs[self.players_ids[0]]["done"] = int(done)
        obs[self.players_ids[1]]["done"] = int(done)
        return obs

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                **self.flat_env.observation_space[self.players_ids[0]],
                "theta": gym.spaces.Box(
                    low=-1, high=1, shape=(self.preference_num,), dtype=np.int64
                ),
                "other_theta": gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(self.preference_num,),
                    dtype=np.int64,
                ),
                "done": gym.spaces.Discrete(2),
            }
        )

    @property
    def action_space(self):
        return self.flat_env.action_space[self.players_ids[0]]

    def set_theta(self, w1, w1_theta):
        self.w1 = w1
        self.thetas = {
            self.players_ids[0]: np.array(
                [0] * (self.preference_num - 1) + [1], dtype=np.int64
            ),
            self.players_ids[1]: w1_theta,
        }
        self.flat_env.set_w(self.w0, self.w1)

    def update_trail_infos(self, infos):
        assert "episode" in infos[self.players_ids[0]]
        ep_game_stats = infos[self.players_ids[0]]["episode"]["ep_game_stats"]

        for i in range(self.num_players):
            self.trail_infos[self.players_ids[i]]["cumulative_sparse_rewards"].append(
                ep_game_stats["cumulative_sparse_rewards_by_agent"][i]
            )
            self.trail_infos[self.players_ids[i]]["cumulative_shaped_rewards"].append(
                ep_game_stats["cumulative_shaped_rewards_by_agent"][i]
            )
            for idx, v in enumerate(
                ep_game_stats["cumulative_category_rewards_by_agent"][i]
            ):
                self.trail_infos[self.players_ids[i]][SHAPED_INFO_KEYS[idx]].append(v)


class ObsWrapper(gym.ObservationWrapper, MultiAgentEnv):
    def __init__(self, env):
        super().__init__(env)
        self.mlp_preprocessor = ModelCatalog.get_preprocessor_for_space(
            self.env.observation_space["done"]
        )
        self._observation_space = self.rewrite_observation_space()

    def rewrite_observation_space(self) -> gym.Space:
        obs_spaces = self.env.observation_space
        obs_spaces["mlp"] = self.mlp_preprocessor.observation_space
        return obs_spaces

    def observation(self, obs):
        obs[self.players_ids[0]]["mlp"] = self.mlp_preprocessor.transform(
            obs[self.players_ids[0]]["done"]
        )
        obs[self.players_ids[1]]["mlp"] = self.mlp_preprocessor.transform(
            obs[self.players_ids[1]]["done"]
        )
        return obs
