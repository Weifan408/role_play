import argparse
import os
import pathlib
import yaml
from datetime import datetime
import numpy as np

import ray
from ray import air
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.evaluation.metrics import summarize_episodes
from ray.tune import registry
from ray.rllib.algorithms.callbacks import make_multi_callbacks

from algorithms.rp2.rp2 import RP, RPConfig
from callbacks.overcooked_callbacks import OvercookedCallbacks
from utils.args import get_cli_args, args_type
from envs.overcooked.Overcooked_Env import Overcooked
from envs.overcooked.meta_rl_wrappers import meta_rl_env, ObsWrapper


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
WANDB_PROJECT = "OVERCOOKED_RP"
WANDB_GROUP = "OVERCOOKED"
WANDB_KEY = "YOURKEY"
EVAL_ITER_TIMES = 4
ENTROPY_COEFF = [
    [0, 0.05],
    [5e7, 0.01],
    [8e7, 0.003],
]


def env_creator(env_config):
    if env_config["env_zoo"] == "overcooked":
        env = Overcooked(
            env_config=env_config,
            run_dir=env_config["run_dir"],
        )
    else:
        raise NotImplementedError
    env = meta_rl_env(env, env_config)
    env = ObsWrapper(env)
    return env


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "rp"


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ray.init(num_cpus=os.cpu_count(), local_mode=args.local)
    log_level = "DEBUG" if args.local else "WARN"
    learner_num_gpus = 0 if args.local else 1
    train_n_replicates = 1 if args.local else 3
    seeds = list(range(train_n_replicates))
    env_config = args.env_config

    # Setup WanDB
    if args.wandb:
        wandb_project = WANDB_PROJECT
        wandb_group = f"{WANDB_GROUP}_{env_config['name']}"

        wdb_callbacks = [
            WandbLoggerCallback(
                project=wandb_project,
                group=wandb_group,
                api_key=WANDB_KEY,
                log_config=True,
                # mode="offline",
            )
        ]
    else:
        wdb_callbacks = []
        print("WARNING! No wandb API key found, running without wandb!")

    registry.register_env(env_config["name"], env_creator)
    test_env = env_creator(env_config)
    # env_config['w'] = tune.grid_search([0.2, 0.3, 0.5])

    rllib_config = (
        RPConfig()
        .environment(
            env_config["name"],
            env_config=env_config,
            # disable_env_checking=True,
        )
        .framework(args.framework)
        .rollouts(**args.rollouts)
        .training(**args.training)
        .multi_agent(
            policies={
                "rp": (
                    None,
                    test_env.observation_space,
                    test_env.action_space,
                    RPConfig.overrides(
                        model={
                            **args.model,
                            "custom_model_config": {
                                **args.custom_model,
                            },
                        },
                    ),
                ),
            },
            algorithm_config_overrides_per_module={
                "rp": RPConfig.overrides(
                    kl_coeff=0.0,
                    use_kl_loss=False,
                    entropy_coeff=ENTROPY_COEFF,
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(
            num_gpus=learner_num_gpus,
            num_gpus_per_learner_worker=learner_num_gpus,
        )
        # .evaluation(
        #     **args.evaluation,
        #     custom_evaluation_function=custom_eval_function,
        # )
        .debugging(
            seed=tune.grid_search(seeds),
            log_level=log_level,
        )
        .callbacks(
            make_multi_callbacks(
                [
                    OvercookedCallbacks,
                ]
            )
        )
        .experimental(
            _disable_preprocessor_api=True,
            _enable_new_api_stack=True,
        )
    )

    tuner = tune.Tuner(
        RP,
        param_space=rllib_config,
        run_config=air.RunConfig(
            name=WANDB_GROUP,
            stop={
                "timesteps_total": args.stop,
            },
            callbacks=wdb_callbacks,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                # num_to_keep=50,
                checkpoint_frequency=1000,
                checkpoint_at_end=True,
            ),
        ),
    )
    results = tuner.fit()
    ray.shutdown()

    assert results.num_errors == 0


if __name__ == "__main__":
    args, remaining = get_cli_args()
    with open(pathlib.Path.cwd() / "configs/overcooked.yaml", "r") as file:
        yaml_content = file.read()
    configs = yaml.safe_load(yaml_content)

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    defaults["local"] = True if args.local else False
    defaults["wandb"] = args.wandb
    defaults["stop"] = args.stop
    defaults["env_config"]["run_dir"] = (
        pathlib.Path.cwd()
        / "results"
        / "rp"
        / "overcooked"
        / defaults["env_config"]["name"]
    )

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))
