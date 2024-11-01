import datetime
from typing import Dict
import gymnasium as gym
import argparse
import os
from collections import OrderedDict
from ml_collections import config_dict
import csv
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.distributions as dist
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from envs.utils import get_default_config
from meltingpot import substrate

from experiments.pretrained.ppo import env_creator
from envs.meltingpot.wrappers.downsampling_obs_wrapper import (
    DownSamplingSubstrateWrapper,
)
from envs.meltingpot.wrappers.meltingpot_wrapper import MeltingPotEnv
from envs.wrappers.meta_rl_wrappers import meta_rl_env, ObsRewriteEnv


# MAX_CYCLES = 200
algorithms = ["brdiv", "trajedi", "inequity"]

checkpoint_path = {
    "rp": {
        "two_player_harvest": "models/baselines/rp/two_player_harvest",
        "two_player_cleanup": "models/baselines/rp/two_player_cleanup",
    },
    "anyplay": {
        "two_player_harvest": "models/baselines/anyplay/two_player_harvest",
        "two_player_cleanup": "models/baselines/anyplay/two_player_cleanup",
    },
    "brdiv": {
        "two_player_harvest": "models/baselines/brdiv/two_player_harvest",
        "two_player_cleanup": "models/baselines/brdiv/two_player_cleanup",
    },
    "trajedi": {
        "two_player_harvest": "models/baselines/trajedi/two_player_harvest",
        "two_player_cleanup": "models/baselines/trajedi/two_player_cleanup",
    },
    "hsp": {
        "two_player_harvest": "models/baselines/hsp/two_player_harvest",
        "two_player_cleanup": "models/baselines/hsp/two_player_cleanup",
    },
}

result_path = {
    "rp": {
        "two_player_harvest": "results/baseline_eval/mp/rp/10_20/two_player_harvest",
        "two_player_cleanup": "results/baseline_eval/mp/rp/10_20/two_player_cleanup",
    },
    "anyplay": {
        "two_player_harvest": "results/baseline_eval/mp/anyplay/two_player_harvest",
        "two_player_cleanup": "results/baseline_eval/mp/anyplay/two_player_cleanup",
    },
    "brdiv": {
        "two_player_harvest": "results/baseline_eval/mp/brdiv/two_player_harvest",
        "two_player_cleanup": "results/baseline_eval/mp/brdiv/two_player_cleanup",
    },
    "trajedi": {
        "two_player_harvest": "results/baseline_eval/mp/trajedi/two_player_harvest",
        "two_player_cleanup": "results/baseline_eval/mp/trajedi/two_player_cleanup",
    },
    "hsp": {
        "two_player_harvest": "results/baseline_eval/mp/hsp/two_player_harvest",
        "two_player_cleanup": "results/baseline_eval/mp/hsp/two_player_cleanup",
    },
}


policy_map = {
    "rp": "rp",
    "selfish": "ppo",
    "collective_reward": "ppo",
    "inequity": "ppo",
}

cleanup_env_config = {
    "env_zoo": "meltingpot",
    "name": "two_player_cleanup",
    "result_path": None,
    "checkpoint_path": None,
    "use_collective_reward": False,
    "use_inequity_reward": False,
    "combine_obs_keys": [
        "RGB",
        "READY_TO_SHOOT",
        "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
        "theta",
        "done",
    ],
}

harvest_env_config = {
    "env_zoo": "meltingpot",
    "name": "two_player_harvest",
    "use_collective_reward": False,
    "use_inequity_reward": False,
    "result_path": None,
    "checkpoint_path": None,
    "combine_obs_keys": ["RGB", "READY_TO_SHOOT", "theta", "done"],
}


def get_env_config(ex_name, algorithm_name):
    if ex_name == "two_player_harvest":
        env_config = harvest_env_config
    elif ex_name == "two_player_cleanup":
        env_config = cleanup_env_config
    else:
        raise NotImplementedError

    env_config["result_path"] = result_path[algorithm_name][ex_name]
    env_config["checkpoint_path"] = checkpoint_path[algorithm_name][ex_name]
    os.makedirs(env_config["result_path"], exist_ok=True)
    return env_config


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex_name", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--pretrained_alg", type=str, required=True)
    parser.add_argument("--checkpoint_num", type=int, default=12)
    args = parser.parse_args()
    return args, parser.parse_known_args()


def process_image(image, scale):
    height, width, _ = image.shape
    if scale:
        image = cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC,
        )

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def experiment(file_path, env_config, role_idx, args, video_dir):
    use_rnn = True
    csv_file = open(file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    device = "cuda:0"
    csv_writer.writerow(
        [
            "episode_k",
            "total_rew_player_0",
            "total_rew_player_1",
            "video_name",
        ]
    )
    if role_idx is not None:
        baseline_model = env_config["checkpoint_path"] + f"/model_{role_idx}_10.pt"
        player_0 = torch.load(baseline_model, device)
        use_rnn = True
    else:  # hsp
        player_0 = torch.load(env_config["checkpoint_path"] + "/arch.pt", device)
        state_dict = torch.load(env_config["checkpoint_path"] + f"/actor.pt")
        player_0.load_state_dict(state_dict)
        player_0.to(device)
        player_0.tpdv["device"] = device
    pretrained_agent_ckpt_path = (
        checkpoint_path[args.pretrained_alg][env_config["name"]]
        + f"/checkpoint_000012/policies/ppo"
    )

    player_1 = Policy.from_checkpoint(pretrained_agent_ckpt_path)  # pretrained model

    env = env_creator(env_config)
    preprocessor = ModelCatalog.get_preprocessor_for_space(
        env.single_agent_observation_space
    )
    # 100次实验
    for i in range(100):
        obs, _ = env.reset()
        total_rewards = [0, 0]
        if use_rnn:
            if role_idx is None:
                hidden0 = torch.zeros(1, 1, 256).to(device)
                masks0 = torch.ones(1, 1).to(device)
            else:
                hidden0 = player_0.init_hidden(1)
        hidden1 = player_1.get_initial_state()

        rgb_frame = env.render()
        video_name = (
            f"{args.baseline}_{args.pretrained_alg}_model{role_idx}_test{i}.webm"
        )
        video_path = os.path.join(video_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(*"vp90")
        height, width, _ = rgb_frame.shape
        out = cv2.VideoWriter(video_path, fourcc, 15, (width * 4, height * 4))
        out.write(process_image(rgb_frame, 4))

        step_cnt = 0
        done = {"__all__": False}
        while not done["__all__"]:
            obs = {
                player_id: preprocessor.transform(obs[player_id]) for player_id in obs
            }
            if role_idx is None:
                action0, act_logits0, hidden0 = player_0(
                    torch.from_numpy(obs["player_0"]).double().unsqueeze(0).to(device),
                    hidden0,
                    masks0,
                )
                action0 = action0.item()
            else:
                act_logits0, hidden0 = player_0(
                    torch.from_numpy(obs["player_0"]).double().unsqueeze(0).to(device),
                    hidden0,
                )
                hidden0 = hidden0.squeeze(0)
                action0 = torch.argmax(act_logits0, dim=-1).item()

            action1, hidden1, _ = player_1.compute_single_action(
                obs["player_1"], state=hidden1
            )
            if args.pretrained_alg != "collective_reward" and action1 == 7:
                action1 = 2

            # print("Actions: ", action0, action1)
            obs, rew, done, _, info = env.step(
                {"player_0": action0, "player_1": action1}
            )
            total_rewards[0] += rew["player_0"]
            total_rewards[1] += rew["player_1"]
            step_cnt += 1
            out.write(process_image(env.render(), 4))

        out.release()
        csv_writer.writerow(
            [
                i,
                total_rewards[0],
                total_rewards[1],
                video_name,
            ]
        )
    csv_file.close()


def experiment2(file_path, env_config, args, video_dir):
    csv_file = open(file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "episode_k",
            "total_rew_player_0",
            "total_rew_player_1",
            "video_name",
        ]
    )

    def env_creator(env_config):
        env_config["other_obs_keys"] = ["theta", "other_theta"]
        env_config = config_dict.ConfigDict(env_config)
        default_config = get_default_config(env_config)
        with default_config.unlocked():
            default_config.update(env_config)
        env = substrate.build_from_config(
            default_config,
            roles=default_config.default_player_roles,
        )
        env = DownSamplingSubstrateWrapper(env)
        env = MeltingPotEnv(env)
        return meta_rl_env(env, env_config)

    env = env_creator(env_config)
    rp_obs_rewrite_env = ObsRewriteEnv(env, env_config)
    baseline_agent_ckpt_path = (
        checkpoint_path[args.baseline][env_config["name"]]
        + f"/checkpoint_0000{args.checkpoint_num}/policies/{policy_map[args.baseline]}"
    )
    pretrained_agent_ckpt_path = (
        checkpoint_path[args.pretrained_alg][env_config["name"]]
        + f"/checkpoint_000012/policies/ppo"
    )

    player_0 = Policy.from_checkpoint(baseline_agent_ckpt_path)
    player_0.model = player_0.model.to("cuda:0")
    player_0.device = "cuda:0"

    player_1 = Policy.from_checkpoint(pretrained_agent_ckpt_path)  # pretrained model
    player_1.model = player_1.model.to("cuda:0")
    player_1.device = "cuda:0"
    player_1_preprocessors = ModelCatalog.get_preprocessor_for_space(
        env.flat_env.single_agent_observation_space
    )
    _ignore_keys = ["theta", "other_theta", "done"]

    for i in range(10):
        obs, info = env.reset()
        hidden0 = player_0.get_initial_state()
        hidden1 = player_1.get_initial_state()
        episode_num = 0
        pre_actions = [None, None]

        rgb_frame = env.render()
        video_name = f"{args.baseline}_{args.pretrained_alg}_role_idx_{np.argmax(env_config['theta'])}_test{i}_episode_{episode_num}.webm"
        video_path = os.path.join(video_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(*"vp90")
        height, width, _ = rgb_frame.shape
        out = cv2.VideoWriter(video_path, fourcc, 15, (width * 4, height * 4))
        out.write(process_image(rgb_frame, 4))

        step_cnt = 0
        done = {"__all__": False}
        while not done["__all__"]:
            obs["player_0"] = rp_obs_rewrite_env.process_obs(obs["player_0"])
            # print(obs["player_0"]["mlp"].shape)
            obs["player_1"] = player_1_preprocessors.transform(
                {
                    key: value
                    for key, value in obs["player_1"].items()
                    if key not in _ignore_keys
                }
            )
            action0, hidden0, _ = player_0.compute_single_action(
                obs["player_0"],
                state=hidden0,
                prev_action=pre_actions[0],
                explore=False,
            )

            action1, hidden1, _ = player_1.compute_single_action(
                obs["player_1"],
                state=hidden1,
                prev_action=pre_actions[1],
                explore=False,
            )
            if args.pretrained_alg != "collective_reward" and action1 == 7:
                action1 = 2

            obs, rew, done, _, info = env.step(
                {"player_0": action0, "player_1": action1}
            )

            pre_actions = [action0, action1]
            step_cnt += 1
            if "__common__" in info:
                hidden1 = player_1.get_initial_state()
                out.release()
                episode_num += 1
                rgb_frame = env.render()
                video_name = f"{args.baseline}_{args.pretrained_alg}_role_idx_{np.argmax(env_config['theta'])}_test{i}_episode_{episode_num}.webm"
                video_path = os.path.join(video_dir, video_name)
                fourcc = cv2.VideoWriter_fourcc(*"vp90")
                height, width, _ = rgb_frame.shape
                out = cv2.VideoWriter(video_path, fourcc, 15, (width * 4, height * 4))
                out.write(process_image(rgb_frame, 4))
            else:
                out.write(process_image(env.render(), 4))

        out.release()
        csv_writer.writerow(
            [
                i,
                info["player_0"]["agent_total_reward"],
                info["player_1"]["agent_total_reward"],
                video_name,
            ]
        )

    csv_file.close()


if __name__ == "__main__":
    args, remaining = get_cli_args()
    pretrain_algs = [
        "selfish",
        "inequity",
        "collective_reward",
    ]
    env_config = get_env_config(args.ex_name, args.baseline)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_path = os.path.join(env_config["result_path"], args.pretrained_alg)
    if args.baseline == "rp":
        for i in range(8):
            one_hot = np.zeros(8)
            one_hot[i] = 1
            env_config["theta"] = one_hot

            file_path = os.path.join(
                result_path,
                f"{current_time}_{args.baseline}_{args.pretrained_alg}_role_idx_{i}.csv",
            )
            video_dir = os.path.join(result_path, "videos")
            os.makedirs(result_path, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            experiment2(file_path, env_config, args, video_dir)
    elif args.baseline == "hsp":
        file_path = os.path.join(
            result_path,
            f"{current_time}_{args.baseline}_{args.pretrained_alg}.csv",
        )
        video_dir = os.path.join(result_path, "videos")
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        experiment(file_path, env_config, None, args, video_dir)
    else:
        for i in range(16):
            file_path = os.path.join(
                result_path,
                f"{current_time}_{args.baseline}_{args.pretrained_alg}_model{i}.csv",
            )
            video_dir = os.path.join(result_path, "videos")
            os.makedirs(result_path, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            experiment(file_path, env_config, i, args, video_dir)
