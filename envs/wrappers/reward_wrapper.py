import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RewardWrapper(gym.core.RewardWrapper, MultiAgentEnv):
    def __init__(self, env, env_config):
        super().__init__(env)
        self.env_config = env_config
        self.alpha = 5.0
        self.beta = 0.05
        self.num_agents = self.env._num_players

    def reward(self, rewards):
        if self.env_config["use_collective_reward"]:
            for agent_id in rewards:
                rewards[agent_id] = sum(rewards.values())
        elif self.env_config["use_inequity_reward"]:
            temp_rewards = rewards.copy()
            for agent in rewards.keys():
                diff = np.array([r - rewards[agent] for r in rewards.values()])
                dis_inequity = self.alpha * sum(diff[diff > 0])
                adv_inequity = self.beta * sum(diff[diff < 0])
                temp_rewards[agent] -= (dis_inequity + adv_inequity) / (
                    self.num_agents - 1
                )
            rewards = temp_rewards
        return rewards
