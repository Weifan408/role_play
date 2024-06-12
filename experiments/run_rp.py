import argparse
import os
import pathlib
import yaml
from datetime import datetime
import importlib
import numpy as np

import ray
from ray import air
from ray import tune
from ray.rllib.policy import policy
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune import registry
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.rllib.algorithms.callbacks import make_multi_callbacks


from algorithms.rp.rp import RPConfig, RP
from callbacks.harvest_callbacks import HarvestCallbacks
from envs.social_dilemmas.harvest_custom import CustomHarvestEnv
from envs.social_dilemmas.cleanup_custom import CustomCleanupEnv
from envs.social_dilemmas.meta_rl_wrappers import meta_rl_env, ObsRewriteEnv
from utils.args import get_cli_args, args_type
from envs.social_dilemmas.maps import TWO_AGENT_HARVEST_MAP, TWO_AGENT_CLEANUP_MAP


cnt = 0
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EVAL_ITER_TIMES = 4

# modify these variables if you want to use wandb
WANDB_PROJECT = "WANDB_PROJECT" 
WANDB_GROUP = "WANDB_GROUP"
YOUR_WANDB_API_KEY = "YOUR_WANDB_API_KEY"


def env_creator(env_config):
    env_config["combine_obs_keys"] = ["curr_obs", "ready_to_shoot", "theta", "done"]
    env_config["other_obs_keys"] = ["theta", "other_theta"]
    if env_config["name"] == "Harvest":
        if env_config["num_agent"] > 2:
            env = CustomHarvestEnv(num_agents=env_config["num_agent"])
        else:
            env = CustomHarvestEnv(ascii_map=TWO_AGENT_HARVEST_MAP, view_size=3, num_agents=2)
    elif env_config["name"] == "CleanUp":
        if env_config["num_agent"] > 2:
            env = CustomCleanupEnv(num_agents=env_config["num_agent"])
        else:
            env = CustomCleanupEnv(ascii_map=TWO_AGENT_CLEANUP_MAP, view_size=3, num_agents=2)
    env = meta_rl_env(env, env_config)
    env = ObsRewriteEnv(env, env_config)
    return env

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "rp"

def custom_eval_function(algorithm, eval_workers):
    theta_idx_list = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 SVOs
    test_theta_idx_list = [
        {
            "agent-0": theta_idx_list[i],
            "agent-1": theta_idx_list[j],
        } for i in range(len(theta_idx_list)) for j in range(i, len(theta_idx_list))
    ]

    rollout_metrics = []
    for eval_theta_idx in test_theta_idx_list:
        eval_workers.foreach_worker(
            func=lambda w: w.foreach_env(
                lambda env: env.set_theta_idx(eval_theta_idx)
            )
        )

        metrics_all_workers = []
        for i in range(EVAL_ITER_TIMES):
            _metrics_all_workers = eval_workers.foreach_worker(
                func=lambda worker: (worker.sample(), worker.get_metrics())[1],
                local_worker=False,
            )
            for metrics_per_worker in _metrics_all_workers:
                metrics_all_workers.extend(metrics_per_worker)

        for metrics_per_worker in metrics_all_workers:
            rollout_metrics.append(metrics_per_worker)

    eval_results = {}
    episodes_num_for_each_test = len(rollout_metrics) // len(test_theta_idx_list)
    for i in range(len(test_theta_idx_list)):
        _metrics = summarize_episodes(
            rollout_metrics[i*episodes_num_for_each_test:(i+1)*episodes_num_for_each_test])

        _metrics.pop("sampler_perf", None)
        _metrics.pop("connector_metrics", None)
        _metrics["policy_reward_max"] = _metrics["policy_reward_max"]["rp"]
        _metrics["policy_reward_mean"] = _metrics["policy_reward_mean"]["rp"]
        _metrics["policy_reward_min"] = _metrics["policy_reward_min"]["rp"]
        eval_results[f"test_{i}"] = _metrics

    return eval_results


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ray.init(num_cpus=os.cpu_count(), local_mode=args.local)
    log_level = "DEBUG" if args.local else "WARN"
    learner_num_gpus = 0 if args.local else 1
    train_n_replicates = 1 if args.local else 1
    seeds = list(range(train_n_replicates))

    # Setup WanDB 
    if args.wandb:
        wandb_project = WANDB_PROJECT
        wandb_group = WANDB_GROUP

        wdb_callbacks = [
            WandbLoggerCallback(
                project=wandb_project,
                group=wandb_group,
                api_key=YOUR_WANDB_API_KEY,
                log_config=True,
                # mode="offline",
            )
        ]
    else:
        wdb_callbacks = []
        print("WARNING! No wandb API key found, running without wandb!")

    env_config = args.env_config
    env_config["num_agent"] = args.num_agent
    
    registry.register_env(env_config["name"], env_creator)
    test_env = env_creator(env_config)
    # env_config["w"] = tune.grid_search([0.2, 0.3, 0.5])

    args.rollouts["num_envs_per_worker"] = EVAL_ITER_TIMES
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
                                "multiplayers": True if env_config["num_agent"] > 2 else False,
                            },
                        },
                    )
                ),
            },
            algorithm_config_overrides_per_module={
                "rp": RPConfig.overrides(
                    kl_coeff=0.0,
                    use_kl_loss=False,
                    entropy_coeff=[
                        [0, 0.1],
                        [5e7, 0.01],
                        [7e7, 0.005],
                    ],
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
        )
        .resources(
            num_gpus=1,
            num_gpus_per_learner_worker=learner_num_gpus,
        )
        .evaluation(
            **args.evaluation,
            custom_evaluation_function = custom_eval_function,
        )
        .debugging(
            seed=tune.grid_search(seeds),
            log_level=log_level,
        )
        .callbacks(make_multi_callbacks([
            HarvestCallbacks,
        ]))
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
                "timesteps_total": 5e8,
            },
            callbacks=wdb_callbacks,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=50,
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
    with open(pathlib.Path.cwd()/"configs/configs.yaml", "r") as file:
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
    defaults["num_agent"] = args.num_agent

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))

