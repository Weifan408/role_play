import gymnasium as gym
from typing import Dict, Optional, Union

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType, PolicyID


class OvercookedCallbacks(DefaultCallbacks):

    def __init__(self):
        super().__init__()

    @override(DefaultCallbacks)
    def on_episode_end(
        self,
        *,
        # TODO (sven): Deprecate Episode/EpisodeV2 with new API stack.
        episode: Union[EpisodeType, Episode, EpisodeV2],
        # TODO (sven): Deprecate this arg new API stack (in favor of `env_runner`).
        worker: Optional["EnvRunner"] = None,
        env_runner: Optional["EnvRunner"] = None,
        # TODO (sven): Deprecate this arg new API stack (in favor of `env`).
        base_env: Optional[BaseEnv] = None,
        env: Optional[gym.Env] = None,
        # TODO (sven): Deprecate this arg new API stack (in favor of `rl_module`).
        policies: Optional[Dict[PolicyID, Policy]] = None,
        rl_module: Optional[RLModule] = None,
        env_index: int,
        **kwargs,
    ) -> None:
        for (agent_id, _), reward in episode.agent_rewards.items():
            trial_info = episode._last_infos[agent_id]["trial"]
            for key, value in trial_info.items():
                episode.custom_metrics[f"{agent_id}_{key}"] = sum(value) / len(value)

            episode.custom_metrics[f"{agent_id}_trail_reward"] = reward
