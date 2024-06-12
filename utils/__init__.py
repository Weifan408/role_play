import functools
import pathlib
import tree

import numpy as np
import torch

from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.nested_dict import NestedDict


action_path = pathlib.Path.cwd().parent / "results" / "actions"


def concat_obs_with_theta(original_obs, player_id, theta):
    processed_obs, filter_obs = {}, {}
    for agent_id, obs in original_obs.items():
        if agent_id == player_id:
            continue
        else:
            filter_obs[agent_id] = obs
            processed_obs[agent_id] = [obs, np.array([theta], dtype=np.float32)]
    return processed_obs, filter_obs  


def clip_method(array):
    tmp_array = array * np.pi
    return np.clip(tmp_array, -np.pi, np.pi)


def set_env_player_policy(algorithm, eval_workers, theta):
    new_pol_id = f"svo_policy_{theta}"

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("low_level_"):
            return "low_level_policy"
        elif agent_id.startswith("high_level_"):
            return "high_level_policy"
        else:
            return new_pol_id

    low_level_state = algorithm.get_policy("low_level_policy").get_state()
    pol_map = eval_workers.local_worker().policy_map

    if new_pol_id in pol_map:
        pol_map[new_pol_id].set_state(low_level_state)
    else:
        inital_policy = algorithm.get_policy("low_level_policy")
        new_policy = algorithm.add_policy(
            policy_id=new_pol_id,
            policy=inital_policy,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train={"high_level_policy", "low_level_policy"},
        )
        new_policy.set_state(low_level_state)
    eval_workers.sync_weights(policies=[new_pol_id])

    def _set(worker):
        worker.set_policy_mapping_fn(policy_mapping_fn)
        if worker.env:
            worker.env.other_theta = theta

    eval_workers.foreach_worker(_set)


def custom_eval_function(algorithm, eval_workers):
    global cnt
    cnt += 1
    actions = []
    svo_theta = np.round(np.linspace(-np.pi, np.pi, 8, endpoint=False), decimals=2)

    for i in range(8):
        set_env_player_policy(algorithm, eval_workers, svo_theta[i])
        # 有缓存问题，所以需要多采样一次
        eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)
        last_samples = eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)
        for idx, env_samples in enumerate(last_samples):
            tmp_actions = {}
            for agent_id, batch in env_samples.policy_batches.items():
                if "low_level" in agent_id:
                    tmp_actions['player_row_action'] = batch['actions']
                elif "high_level" in agent_id:
                    tmp_actions['high_level_agent_action'] = np.round(clip_method(np.array(batch['actions']).flatten()), decimals=2)
                else:
                    tmp_actions[agent_id] = batch["actions"]
            actions.append(tmp_actions)
    episodes = collect_episodes(workers=eval_workers, timeout_seconds=99999)
    metrics = summarize_episodes(episodes)
    eval_action_path = (action_path /
                        "eval_actions_iter_{}.npy".format(cnt))
    np.save(eval_action_path, np.array(actions))

    return metrics


def tokenize(tokenizer, inputs: dict, framework: str) -> dict:
    """Tokenizes the observations from the input dict.

    Args:
        tokenizer: The tokenizer to use.
        inputs: The input dict.

    Returns:
        The output dict.
    """
    # Tokenizer may depend solely on observations.
    obs = inputs[SampleBatch.OBS]
    tokenizer_inputs = {SampleBatch.OBS: obs}
    if isinstance(obs, dict) or isinstance(obs, NestedDict):
        size = list(obs.values())[0].size() if framework == "torch" else list(
            obs.values()
        )[0].shape
    else:
        size = list(obs.size() if framework == "torch" else obs.shape)

    b_dim, t_dim = size[:2]
    fold, unfold = get_fold_unfold_fns(b_dim, t_dim, framework=framework)
    # Push through the tokenizer encoder.
    out = tokenizer(fold(tokenizer_inputs))
    out = out[ENCODER_OUT]
    # Then unfold batch- and time-dimensions again.
    return unfold(out)


def get_fold_unfold_fns(b_dim: int, t_dim: int, framework: str):
    if framework in "tf2":
        raise NotImplementedError("tf2 not implemented yet!")
    elif framework == "torch":

        def fold_mapping(item):
            if item is None:
                # Torch has no representation for `None`, so we return None
                return item
            item = torch.as_tensor(item)
            size = list(item.size())
            current_b_dim, current_t_dim = list(size[:2])

            assert (b_dim, t_dim) == (current_b_dim, current_t_dim), (
                "All tensors in the struct must have the same batch and time "
                "dimensions. Got {} and {}.".format(
                    (b_dim, t_dim), (current_b_dim, current_t_dim)
                )
            )

            other_dims = size[2:]
            return item.reshape([b_dim * t_dim] + other_dims)

        def unfold_mapping(item):
            if item is None:
                return item
            item = torch.as_tensor(item)
            size = list(item.size())
            current_b_dim = size[0]
            other_dims = size[1:]
            assert current_b_dim == b_dim * t_dim, (
                "The first dimension of the tensor must be equal to the product of "
                "the desired batch and time dimensions. Got {} and {}.".format(
                    current_b_dim, b_dim * t_dim
                )
            )
            return item.reshape([b_dim, t_dim] + other_dims)

    else:
        raise ValueError(f"framework {framework} not implemented!")

    return functools.partial(tree.map_structure, fold_mapping), functools.partial(
        tree.map_structure, unfold_mapping
    )

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def two_hot(value, num_classes, lower_bound=-np.pi, upper_bound=np.pi):
    value = np.clip(value, lower_bound, upper_bound)
    value = np.atleast_1d(value)

    class_delta = (upper_bound - lower_bound) / (num_classes - 1)
    idx = (-lower_bound + value) / class_delta

    k = np.floor(idx)
    kp1 = np.ceil(idx)
    kp1 = np.where(k == kp1, kp1 + 1.0, kp1)
    kp1 = np.where(kp1 == num_classes, kp1 - 2.0, kp1)

    values_k = lower_bound + k * class_delta
    values_kp1 = lower_bound + kp1 * class_delta

    weights_k = (value - values_kp1) / (values_k - values_kp1)
    weights_kp1 = 1.0 - weights_k

    indices_k = np.stack([np.arange(value.shape[0]), k.astype(int)], -1)
    indices_kp1 = np.stack([np.arange(value.shape[0]), kp1.astype(int)], -1)
    indices = np.concatenate([indices_k, indices_kp1], 0)

    updates = np.concatenate([weights_k, weights_kp1], 0)

    out = np.zeros((value.shape[0], num_classes))
    out[indices[:, 0], indices[:, 1]] = updates

    return out

