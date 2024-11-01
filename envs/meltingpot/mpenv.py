import cv2
import importlib

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from meltingpot import substrate
from envs.social_dilemmas.utility_funcs import make_video_from_rgb_imgs

from envs.meltingpot import utils


class MeltingPotEnv(MultiAgentEnv):
    def __init__(self, config):
        self.config = config
        self.view_size = config.get("view_size", 7)
        self.substrate_name = config["name"]
        default_config = self.get_default_config(self.substrate_name)
        with default_config.unlocked():
            default_config.update(config)
        self.mp_env = utils.parallel_env(default_config)
        self.num_agents = self.mp_env.num_agents
        self.player_ids = self.mp_env.agents

    def reset(self, seed=None, options=None):
        obs, info = self.mp_env.reset(seed, options)
        return self.process_obs(obs), info

    def step(self, actions):
        obs, rewards, dones, info = self.mp_env.step(actions)
        return self.process_obs(obs), rewards, dones, info

    def render(self, mode="human", filename=None):
        return self.mp_env.render(mode, filename)

    def process_obs(self, obs):
        for player_id in obs:
            obs[player_id]["RGB"] = cv2.resize(
                obs[player_id]["RGB"],
                (self.view_size, self.view_size),
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
