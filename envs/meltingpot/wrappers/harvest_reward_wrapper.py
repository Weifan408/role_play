import gymnasium as gym

from ray.rllib.env.multi_agent_env import MultiAgentEnv

BeingZapped_REWARD = -10
Zapping_REWARD = -1
EATAPPLE_REWARD = 1


class HarvestRewardWrapper(gym.core.RewardWrapper, MultiAgentEnv):

    def __init__(self, env):
        super().__init__(env)
        assert (
            self.env.w0 is not None
        ), "MeltingPotEnv requires the environment to have a 'w0' attribute"
        assert (
            self.env.w1 is not None
        ), "MeltingPotEnv requires the environment to have a 'w1' attribute"
        self._agent_ids = self.env._agent_ids
        self.beingzapped_count = {agent_id: 0 for agent_id in self._agent_ids}
        self.zapping_count = {agent_id: 0 for agent_id in self._agent_ids}
        self.harvest_count = {agent_id: 0 for agent_id in self._agent_ids}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed, options)
        self.beingzapped_count = {agent_id: 0 for agent_id in self._agent_ids}
        self.zapping_count = {agent_id: 0 for agent_id in self._agent_ids}
        self.harvest_count = {agent_id: 0 for agent_id in self._agent_ids}

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        for agent_id in self._agent_ids:
            info[agent_id]["beingzapped_count"] = self.beingzapped_count[agent_id]
            info[agent_id]["zapping_count"] = self.zapping_count[agent_id]
            info[agent_id]["harvest_count"] = self.harvest_count[agent_id]
        return observation, self.reward(reward), terminated, truncated, info

    def reward(self, rewards):
        hidden_rewards = {agent_id: 0 for agent_id in self._agent_ids}

        # BeingZapped
        if rewards[self._agent_ids[0]] < -1:
            hidden_rewards[self._agent_ids[0]] += self.w0[0]
            self.beingzapped_count[self._agent_ids[0]] += 1
        if rewards[self._agent_ids[1]] < -1:
            hidden_rewards[self._agent_ids[1]] += self.w1[0]
            self.beingzapped_count[self._agent_ids[1]] += 1

        # Zapping
        if rewards[self._agent_ids[0]] == -1 or rewards[self._agent_ids[0]] == -11:
            hidden_rewards[self._agent_ids[0]] += self.w0[1]
            self.zapping_count[self._agent_ids[0]] += 1
        if rewards[self._agent_ids[1]] == -1 or rewards[self._agent_ids[1]] == -11:
            hidden_rewards[self._agent_ids[1]] += self.w1[1]
            self.zapping_count[self._agent_ids[1]] += 1

        # EatApple
        if rewards[self._agent_ids[0]] == 1 or rewards[self._agent_ids[0]] == -9:
            hidden_rewards[self._agent_ids[0]] += self.w0[2]
            self.harvest_count[self._agent_ids[0]] += 1
        if rewards[self._agent_ids[1]] == 1 or rewards[self._agent_ids[1]] == -9:
            hidden_rewards[self._agent_ids[1]] += self.w1[2]
            self.harvest_count[self._agent_ids[1]] += 1

        rewards[self._agent_ids[0]] = (
            self.w0[3] * rewards[self._agent_ids[0]]
            + hidden_rewards[self._agent_ids[0]]
        )
        rewards[self._agent_ids[1]] = (
            self.w1[3] * rewards[self._agent_ids[1]]
            + hidden_rewards[self._agent_ids[1]]
        )

        return rewards
