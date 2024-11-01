# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for Clean Up.

Example video: https://youtu.be/TqiJYxOwdxw

Clean Up is a seven player game. Players are rewarded for collecting apples. In
Clean Up, apples grow in an orchard at a rate inversely related to the
cleanliness of a nearby river. The river accumulates pollution at a constant
rate. The apple growth rate in the orchard drops to zero once the pollution
accumulates past a threshold value. Players have an additional action allowing
them to clean a small amount of pollution from the river in front of themselves.
They must physically leave the apple orchard to clean the river. Thus, players
must maintain a public good of high orchard regrowth rate through effortful
contributions. This is a public good provision problem because the benefit of a
healthy orchard is shared by all, but the costs incurred to ensure it exists are
born by individuals.

Players are also able to zap others with a beam that removes any player hit by
it from the game for 50 steps.

Clean Up was first described in Hughes et al. (2018).

Hughes, E., Leibo, J.Z., Phillips, M., Tuyls, K., Duenez-Guzman, E.,
Castaneda, A.G., Dunning, I., Zhu, T., McKee, K., Koster, R. and Roff, H., 2018,
Inequity aversion improves cooperation in intertemporal social dilemmas. In
Proceedings of the 32nd International Conference on Neural Information
Processing Systems (pp. 3330-3340).
"""

from typing import Any, Dict, Mapping, Sequence

from ml_collections import config_dict

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import game_object_utils
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs

from meltingpot.configs.substrates.clean_up import (
    human_readable_colors,
    ACTION_SET,
    create_prefabs,
    CHAR_PREFAB_MAP,
    TARGET_SPRITE_SELF,
)

PrefabConfig = game_object_utils.PrefabConfig
TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": True,
}

# Warning: setting `_ENABLE_DEBUG_OBSERVATIONS = True` may cause slowdown.
_ENABLE_DEBUG_OBSERVATIONS = False
_COMPASS = ["N", "E", "S", "W"]


ASCII_MAP = """
WWWWWWWWWWWWW
WFFFFF P BBBW
WHHHH   BBBBW
WFFFFFSSSBBBW
WHHHH   BBBBW
WFFFFF P BBBW
WWWWWWWWWWWWW
"""


def create_scene():
    """Create the scene object, a non-physical object to hold global logic."""
    scene = {
        "name": "scene",
        "components": [
            {
                "component": "StateManager",
                "kwargs": {
                    "initialState": "scene",
                    "stateConfigs": [
                        {
                            "state": "scene",
                        }
                    ],
                },
            },
            {
                "component": "Transform",
            },
            {
                "component": "RiverMonitor",
                "kwargs": {},
            },
            {
                "component": "DirtSpawner",
                "kwargs": {
                    "dirtSpawnProbability": 0.5,
                    "delayStartOfDirtSpawning": 50,
                },
            },
            {
                "component": "StochasticIntervalEpisodeEnding",
                "kwargs": {
                    "minimumFramesPerEpisode": 200,
                    "intervalLength": 10,  # Set equal to unroll length.
                    "probabilityTerminationPerInterval": 0.2,
                },
            },
            {
                "component": "GlobalData",
            },
        ],
    }
    return scene


def create_avatar_object(
    player_idx: int, target_sprite_self: Dict[str, Any]
) -> Dict[str, Any]:
    """Create an avatar object that always sees itself as blue."""
    # Lua is 1-indexed.
    lua_index = player_idx + 1

    # Setup the self vs other sprite mapping.
    source_sprite_self = "Avatar" + str(lua_index)
    custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}

    live_state_name = "player{}".format(lua_index)
    avatar_object = {
        "name": f"avatar{lua_index}",
        "components": [
            {
                "component": "StateManager",
                "kwargs": {
                    "initialState": live_state_name,
                    "stateConfigs": [
                        # Initial player state.
                        {
                            "state": live_state_name,
                            "layer": "superOverlay",
                            "sprite": source_sprite_self,
                            "contact": "avatar",
                            "groups": ["players"],
                        },
                        # Player wait type for times when they are zapped out.
                        {"state": "playerWait", "groups": ["playerWaits"]},
                    ],
                },
            },
            {
                "component": "Transform",
            },
            {
                "component": "Appearance",
                "kwargs": {
                    "renderMode": "ascii_shape",
                    "spriteNames": [source_sprite_self],
                    "spriteShapes": [shapes.CUTE_AVATAR],
                    "palettes": [shapes.get_palette(human_readable_colors[player_idx])],
                    "noRotates": [True],
                },
            },
            {
                "component": "AdditionalSprites",
                "kwargs": {
                    "renderMode": "ascii_shape",
                    "customSpriteNames": [target_sprite_self["name"]],
                    "customSpriteShapes": [target_sprite_self["shape"]],
                    "customPalettes": [target_sprite_self["palette"]],
                    "customNoRotates": [target_sprite_self["noRotate"]],
                },
            },
            {
                "component": "Avatar",
                "kwargs": {
                    "index": lua_index,
                    "aliveState": live_state_name,
                    "waitState": "playerWait",
                    "spawnGroup": "spawnPoints",
                    "actionOrder": ["move", "turn", "fireZap", "fireClean"],
                    "actionSpec": {
                        "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                        "turn": {"default": 0, "min": -1, "max": 1},
                        "fireZap": {"default": 0, "min": 0, "max": 1},
                        "fireClean": {"default": 0, "min": 0, "max": 1},
                    },
                    "view": {
                        "left": 3,
                        "right": 3,
                        "forward": 3,
                        "backward": 3,
                        "centered": True,
                    },
                    "spriteMap": custom_sprite_map,
                },
            },
            {
                "component": "Zapper",
                "kwargs": {
                    "cooldownTime": 5,
                    "beamLength": 2,
                    "beamRadius": 1,
                    "framesTillRespawn": 25,
                    "penaltyForBeingZapped": -10,
                    "rewardForZapping": -1,
                    # "removeHitPlayer": True,
                },
            },
            {
                "component": "ReadyToShootObservation",
            },
            {
                "component": "Cleaner",
                "kwargs": {
                    "cooldownTime": 2,
                    "beamLength": 2,
                    "beamRadius": 1,
                },
            },
            {
                "component": "Taste",
                "kwargs": {
                    "role": "free",
                    "rewardAmount": 1,
                },
            },
            {
                "component": "AllNonselfCumulants",
            },
        ],
    }
    # Signals needed for puppeteers.
    metrics = [
        {
            "name": "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
            "type": "Doubles",
            "shape": [],
            "component": "AllNonselfCumulants",
            "variable": "num_others_who_cleaned_this_step",
        },
    ]
    if _ENABLE_DEBUG_OBSERVATIONS:
        avatar_object["components"].append(
            {
                "component": "LocationObserver",
                "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
            }
        )
        # Debug metrics
        metrics.append(
            {
                "name": "PLAYER_CLEANED",
                "type": "Doubles",
                "shape": [],
                "component": "Cleaner",
                "variable": "player_cleaned",
            }
        )
        metrics.append(
            {
                "name": "PLAYER_ATE_APPLE",
                "type": "Doubles",
                "shape": [],
                "component": "Taste",
                "variable": "player_ate_apple",
            }
        )
        metrics.append(
            {
                "name": "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP",
                "type": "Doubles",
                "shape": [],
                "component": "Zapper",
                "variable": "num_others_player_zapped_this_step",
            }
        )
        metrics.append(
            {
                "name": "NUM_OTHERS_WHO_ATE_THIS_STEP",
                "type": "Doubles",
                "shape": [],
                "component": "AllNonselfCumulants",
                "variable": "num_others_who_ate_this_step",
            }
        )

    # Add the metrics reporter.
    avatar_object["components"].append(
        {"component": "AvatarMetricReporter", "kwargs": {"metrics": metrics}}
    )

    return avatar_object


def create_avatar_objects(num_players):
    """Returns list of avatar objects of length 'num_players'."""
    avatar_objects = []
    for player_idx in range(0, num_players):
        game_object = create_avatar_object(player_idx, TARGET_SPRITE_SELF)
        avatar_objects.append(game_object)

    return avatar_objects


def get_config():
    """Default configuration for the clean_up level."""
    config = config_dict.ConfigDict()

    # Action set configuration.
    config.action_set = ACTION_SET
    # Observation format configuration.
    config.individual_observation_names = [
        "RGB",
        "READY_TO_SHOOT",
        # Global switching signals for puppeteers.
        "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
        "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP",
        "PLAYER_ATE_APPLE",
        "PLAYER_CLEANED",
    ]
    config.global_observation_names = [
        "WORLD.RGB",
    ]

    # The specs of the environment (from a single-agent perspective).
    config.action_spec = specs.action(len(ACTION_SET))
    config.timestep_spec = specs.timestep(
        {
            "RGB": specs.OBSERVATION["RGB"],
            "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
            # Global switching signals for puppeteers.
            "NUM_OTHERS_WHO_CLEANED_THIS_STEP": specs.float64(),
            "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP": specs.float64(),
            "PLAYER_ATE_APPLE": specs.float64(),
            "PLAYER_CLEANED": specs.float64(),
            # Debug only (do not use the following observations in policies).
            "WORLD.RGB": specs.rgb(168, 240),
        }
    )

    # The roles assigned to each player.
    config.valid_roles = frozenset({"default"})
    config.default_player_roles = ("default",) * 2

    return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
    """Build the clean_up substrate given roles."""
    del config
    num_players = len(roles)
    # Build the rest of the substrate definition.
    substrate_definition = dict(
        levelName="clean_up",
        levelDirectory="meltingpot/lua/levels",
        numPlayers=num_players,
        # Define upper bound of episode length since episodes end stochastically.
        maxEpisodeLengthFrames=5000,
        spriteSize=8,
        topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
        simulation={
            "map": ASCII_MAP,
            "gameObjects": create_avatar_objects(num_players),
            "scene": create_scene(),
            "prefabs": create_prefabs(),
            "charPrefabMap": CHAR_PREFAB_MAP,
        },
    )
    return substrate_definition
