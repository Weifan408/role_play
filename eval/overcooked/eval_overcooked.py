import argparse
import csv
import pathlib
import yaml

import numpy as np
import pandas as pd
from ray.rllib.policy.policy import Policy

import torch

from envs.overcooked.Overcooked_Env import Overcooked
from envs.overcooked.meta_rl_wrappers import meta_rl_env, ObsWrapper

NUM_CLASS = 3


def env_creator(env_config):
    if env_config["env_zoo"] == "overcooked":
        from envs.overcooked.Overcooked_Env import Overcooked

        env = Overcooked(
            env_config=env_config,
            run_dir=env_config["run_dir"],
        )
    else:
        raise NotImplementedError
    env = meta_rl_env(env, env_config)
    env = ObsWrapper(env)
    return env


def experiment(env_config, alg_name, checkpoint_num, device):
    env = env_creator(env_config)
    env_name = env_config["env_zoo"]
    layout_name = env_config["name"]
    prev_action, prev_reward = None, None
    agent_0 = Policy.from_checkpoint(
        f"models/{env_name}/{alg_name}/{layout_name}/checkpoint_{checkpoint_num:06d}/policies/{alg_name}"
    )
    agent_0.model = agent_0.model.to(device)
    agent_0.device = device
    done = {}
    results = []
    for i in range(10):  # 10 trails = 100 episodes
        obs, info = env.reset()
        hidden0 = agent_0.get_initial_state()
        total_rewards = [[], []]
        predictions = []
        done["__all__"] = False
        while not done["__all__"]:
            obs_alg = obs[f"agent_0"]

            action0, hidden0, out = agent_0.compute_single_action(
                obs_alg,
                state=hidden0,
                prev_action=prev_action,
                # prev_reward=prev_reward,
                explore=False,
            )
            tmp = out["predicted_out_one_hot_sample"].reshape(-1, NUM_CLASS)
            predict = np.argmax(tmp, axis=-1) - 1
            predictions.append(predict)
            prev_action = action0
            actions = {f"agent_0": action0}

            obs, rew, done, _, info = env.step(actions)
            # prev_reward = rew["agent_0"]

            if "episode" in info:
                episode_info = info["episode"]
                total_rewards[0].append(
                    episode_info["ep_sparse_r"]
                    + episode_info["ep_shaped_r_by_agent"][0]
                )
                total_rewards[1].append(
                    episode_info["ep_sparse_r"]
                    + episode_info["ep_shaped_r_by_agent"][1]
                )

        results.append(
            {
                "trial": i,
                "total_rewards_agent_0": total_rewards[0],
                "total_rewards_agent_1": total_rewards[1],
                "predictions": predictions,
                **info[f"agent_0"]["trial"],
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(f"{env_config['run_dir']}/experiment_results.csv", index=False)


def experiment2(env_config, alg_name, device="cpu"):
    env = Overcooked(
        env_config=env_config,
        run_dir=env_config["run_dir"],
    )
    env_name = env_config["env_zoo"]
    layout_name = env_config["name"]

    agent_0 = torch.load(
        f"models/{env_name}/{alg_name}/{layout_name}/actor.pt",
        device,
    )
    agent_0.tpdv["device"] = device

    done = {}
    results = []
    for i in range(100):
        obs, info = env.reset()
        total_rewards = [0, 0]
        episodes_rewards = {}

        hidden0 = torch.zeros(1, 1, 256).to(device)
        masks0 = torch.ones(1, 1).to(device)

        done["__all__"] = False
        while not done["__all__"]:
            tmp = (
                torch.Tensor(obs[f"agent_0"]["featurize_obs"])
                .to(torch.float64)
                .to(device)
            )
            tmp = tmp.unsqueeze(0)
            action0, act_logits0, hidden0 = agent_0(
                tmp,
                hidden0,
                masks0,
            )
            action0 = action0.item()
            actions = [action0, None]
            obs, rew, done, _, info = env.step(actions)

            if "episode" in info["agent_0"]:
                episodes_rewards["sparse_reward"] = info["agent_0"]["episode"][
                    "ep_sparse_r_by_agent"
                ][0]
                episodes_rewards["shaped_rewrad"] = info["agent_0"]["episode"][
                    "ep_shaped_r_by_agent"
                ][0]
            total_rewards[0] += rew["agent_0"]
            total_rewards[1] += rew["agent_1"]

        results.append(
            {
                "episode": i,
                "total_rewards_agent_0": total_rewards[0],
                "total_rewards_agent_1": total_rewards[1],
                "sparse_reward_agent_0": episodes_rewards["sparse_reward"],
                "shaped_reward_agent_0": episodes_rewards["shaped_rewrad"],
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(f"{env_config['run_dir']}/experiment_results.csv", index=False)


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--script_agent", type=str, default=None)
    parser.add_argument("--alg_name", type=str, default="rp")
    parser.add_argument("--checkpoint_num", type=int, default=12)
    parser.add_argument("--device", str, default="cpu")
    args = parser.parse_args()
    return args, parser.parse_known_args()


if __name__ == "__main__":
    args, remaining = get_cli_args()

    with open(pathlib.Path.cwd() / "configs/overcooked.yaml", "r") as file:
        yaml_content = file.read()
    configs = yaml.safe_load(yaml_content)

    env_config = configs["eval"][args.layout]
    env_config["script_agent"] = args.script_agent
    env_config["add_hidden_reward"] = False

    if args.alg_name == "rp":
        env_config["run_dir"] = (
            pathlib.Path.cwd()
            / "results"
            / "overcooked"
            / args.layout
            / args.script_agent[7:]
            / args.alg_name
        )
        experiment(
            env_config,
            args.alg_name,
            checkpoint_num=args.checkpoint_num,
        )
    else:
        env_config["run_dir"] = (
            pathlib.Path.cwd()
            / "results"
            / "overcooked"
            / args.layout
            / args.script_agent[7:]
            / args.alg_name
        )
        experiment2(env_config, args.alg_name, args.device)
