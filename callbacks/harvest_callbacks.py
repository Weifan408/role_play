import collections
from datetime import datetime
from functools import partial
import pathlib
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from ray.rllib.algorithms.algorithm import Algorithm

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID


from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override


import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
    override,
    OverrideToImplementCustomLogic,
    PublicAPI,
)
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.exploration.random_encoder import (
    _MovingMeanStd,
    compute_states_entropy,
    update_beta,
)
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID
from ray.tune.callback import _CallbackMeta

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker


class HarvestCallbacks(DefaultCallbacks):
    

    def __init__(self):
        super().__init__()
        self.train_num = 0
        
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        pass

    def on_epsidoe_step(
        self, *, 
        worker: RolloutWorker, 
        base_env: BaseEnv, 
        episode: Episode, 
        env_index: int, 
        **kwargs
    ):
        pass
    
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
            episode.custom_metrics[f"real_reward_{agent_id}"] = sum(episode._last_infos[agent_id]["agent_total_reward"]) / len(episode._last_infos[agent_id]["agent_total_reward"])
            episode.custom_metrics[f"last_episode_reward_{agent_id}"] = episode._last_infos[agent_id]["agent_total_reward"][-1]

            episode.custom_metrics["all_agents_total_reward"] = sum(episode._last_infos[agent_id]["all_agents_total_reward"]) / len(episode._last_infos[agent_id]["all_agents_total_reward"])
            episode.custom_metrics["last_episode_all_agents_total_reward"] = episode._last_infos[agent_id]["all_agents_total_reward"][-1]

            # episode.custom_metrics[f"theta_dis_{agent_id}"] = sum(episode._last_infos[agent_id]["theta_dis"]) / len(episode._last_infos[agent_id]["theta_dis"])
            # episode.custom_metrics[f"last_episode_theta_dis_{agent_id}"] = episode._last_infos[agent_id]["theta_dis"][-1]
            episode.custom_metrics[f"avg_compute_total_reward_{agent_id}"] = sum(episode._last_infos[agent_id]["compute_total_reward"]) / len(episode._last_infos[agent_id]["compute_total_reward"])
            episode.custom_metrics[f"last_episode_compute_total_reward_{agent_id}"] = episode._last_infos[agent_id]["compute_total_reward"][-1] 
            # episode.custom_metrics[f"theta_{agent_id}"] = worker.env.env.theta_map[worker.env.env.theta_idx[agent_id]]
            # episode.custom_metrics['real_theta'] = episode._last_infos[agent_id]["real_theta"][-1]

    @override(DefaultCallbacks)
    def on_train_result(self, *, algorithm: Algorithm, result: Dict, **kwargs) -> None:
        self.train_num += 1
        if self.train_num < 500:
            phase = 0
        else:
            phase = 1

        def _set_phase(worker, phase):
            if worker.env:
                worker.foreach_env(lambda env: env.set_phase(phase))
        
        algorithm.workers.foreach_worker(partial(_set_phase, phase=phase))

    @override(DefaultCallbacks)
    def on_evaluate_start(
        self, 
        *, 
        algorithm: Algorithm, 
        **kwargs
    ) -> None:
        return super().on_evaluate_start(algorithm=algorithm, **kwargs)

class HarvestCallbacks2(DefaultCallbacks):
    

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
            episode.custom_metrics[f"real_reward_{agent_id}"] = episode._last_infos[agent_id]["agent_total_reward"]
            episode.custom_metrics[f"last_episode_reward_{agent_id}"] = episode._last_infos[agent_id]["agent_total_reward"]

            episode.custom_metrics["all_agents_total_reward"] = episode._last_infos[agent_id]["all_agents_total_reward"]
            episode.custom_metrics["last_episode_all_agents_total_reward"] = episode._last_infos[agent_id]["all_agents_total_reward"]

            episode.custom_metrics[f"avg_compute_total_reward_{agent_id}"] = episode._last_infos[agent_id]["compute_total_reward"]
            episode.custom_metrics[f"last_episode_compute_total_reward_{agent_id}"] = episode._last_infos[agent_id]["compute_total_reward"]

    @override(DefaultCallbacks)
    def on_train_result(self, *, algorithm: Algorithm, result: Dict, **kwargs) -> None:
        self.train_num += 1
        if self.train_num < 500:
            phase = 0
        else:
            phase = 1

        def _set_phase(worker, phase):
            if worker.env:
                worker.foreach_env(lambda env: env.set_phase(phase))
        
        algorithm.workers.foreach_worker(partial(_set_phase, phase=phase))
