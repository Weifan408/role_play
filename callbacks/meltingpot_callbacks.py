from functools import partial
from typing import Dict, Optional, Union

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import PolicyID


class MeltingPotCallbacks(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.train_num = 0

    @override(DefaultCallbacks)
    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        for (agent_id, _), reward in episode.agent_rewards.items():
            episode.custom_metrics[f"real_reward_{agent_id}"] = sum(
                episode._last_infos[agent_id]["agent_total_reward"]
            ) / len(episode._last_infos[agent_id]["agent_total_reward"])
            episode.custom_metrics[f"last_episode_reward_{agent_id}"] = (
                episode._last_infos[agent_id]["agent_total_reward"][-1]
            )

            episode.custom_metrics["all_agents_total_reward"] = sum(
                episode._last_infos[agent_id]["all_agents_total_reward"]
            ) / len(episode._last_infos[agent_id]["all_agents_total_reward"])
            episode.custom_metrics["last_episode_all_agents_total_reward"] = (
                episode._last_infos[agent_id]["all_agents_total_reward"][-1]
            )

            episode.custom_metrics[f"avg_compute_total_reward_{agent_id}"] = sum(
                episode._last_infos[agent_id]["compute_total_reward"]
            ) / len(episode._last_infos[agent_id]["compute_total_reward"])
            episode.custom_metrics[f"last_episode_compute_total_reward_{agent_id}"] = (
                episode._last_infos[agent_id]["compute_total_reward"][-1]
            )

    @override(DefaultCallbacks)
    def on_train_result(self, *, algorithm: Algorithm, result: Dict, **kwargs) -> None:
        self.train_num += 1
        phase = 1
        # if self.train_num < 200:
        #     phase = 0
        # else:
        #     phase = 1

        def _set_phase(worker, phase):
            if worker.env:
                worker.foreach_env(lambda env: env.set_phase(phase))

        algorithm.workers.foreach_worker(partial(_set_phase, phase=phase))
