import importlib
import numpy as np


def obs_transform(obs, theta_idx, done=None):
    player_ids = list(obs.keys())

    for player_id in obs:
        specific_obs = obs[player_id]
        if isinstance(specific_obs, dict):
            obs[player_id] = {**specific_obs}
        else:
            obs[player_id]["obs"] = specific_obs
        #
        obs[player_id]["theta"] = theta_idx[player_id]
        other_player_id = [id for id in player_ids if id != player_id]
        obs[player_id]["other_theta"] = [theta_idx[id] for id in other_player_id]
        if done is not None:
            obs[player_id]["done"] = done
    return obs


def get_default_config(env_config):
    substrate = env_config["name"]
    if substrate.startswith("two_player"):
        path = f"envs.meltingpot.configs"
    else:
        path = f"meltingpot.configs.substrates"
    module = importlib.import_module(path)
    return module.get_config(substrate)
