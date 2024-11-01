import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class ActionWrapper(gym.core.ActionWrapper, MultiAgentEnv):
    # def __init__(self, env):
    #     super().__init__(env)

    def action(self, actions):
        for agent_id, action in actions.items():
            actions[agent_id] = action if action != 7 else 2
        return actions
