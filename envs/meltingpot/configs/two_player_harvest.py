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
"""Configuration for Commons Harvest: Open.

Example video: https://youtu.be/lZ-qpPP4BNE

Apples are spread around the map and can be consumed for a reward of 1. Apples
that have been consumed regrow with a per-step probability that depends on the
number of uneaten apples in a `L2` norm neighborhood of radius 2 (by default).
After an apple has been eaten and thus removed, its regrowth probability depends
on the number of uneaten apples still in its local neighborhood. With standard
parameters, it the grown rate decreases as the number of uneaten apples in the
neighborhood decreases and when there are zero uneaten apples in the
neighborhood then the regrowth rate is zero. As a consequence, a patch of apples
that collectively doesn't have any nearby apples, can be irrevocably lost if all
apples in the patch are consumed. Therefore, agents must exercise restraint when
consuming apples within a patch. Notice that in a single agent situation, there
is no incentive to collect the last apple in a patch (except near the end of the
episode). However, in a multi-agent situation, there is an incentive for any
agent to consume the last apple rather than risk another agent consuming it.
This creates a tragedy of the commons from which the substrate derives its name.

This mechanism was first described in Janssen et al (2010) and adapted for
multi-agent reinforcement learning in Perolat et al (2017).

Janssen, M.A., Holahan, R., Lee, A. and Ostrom, E., 2010. Lab experiments for
the study of social-ecological systems. Science, 328(5978), pp.613-617.

Perolat, J., Leibo, J.Z., Zambaldi, V., Beattie, C., Tuyls, K. and Graepel, T.,
2017. A multi-agent reinforcement learning model of common-pool
resource appropriation. In Proceedings of the 31st International Conference on
Neural Information Processing Systems (pp. 3646-3655).
"""

from typing import Any, Dict, Mapping, Sequence

from meltingpot.utils.substrates import colors
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import specs

from meltingpot.configs.substrates.commons_harvest__open import (
    APPLE_RESPAWN_RADIUS,
    REGROWTH_PROBABILITIES,
    CHAR_PREFAB_MAP,
    create_prefabs,
    ACTION_SET,
    TARGET_SPRITE_SELF,
    # create_avatar_objects,
)

from ml_collections import config_dict
import numpy as np


ASCII_MAP = """
WWWWWWWWWWWWW
W           W
W    AAA    W
W P AAAAA P W
W    AAA    W
W           W
WWWWWWWWWWWWW
"""

_ENABLE_DEBUG_OBSERVATIONS = False
_COMPASS = ["N", "E", "S", "W"]


def create_scene():
    """Creates the scene with the provided args controlling apple regrowth."""
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
            {"component": "Neighborhoods", "kwargs": {}},
            {
                "component": "StochasticIntervalEpisodeEnding",
                "kwargs": {
                    "minimumFramesPerEpisode": 200,
                    "intervalLength": 10,  # Set equal to unroll length.
                    "probabilityTerminationPerInterval": 0.15,
                },
            },
        ],
    }

    return scene


def create_avatar_object(
    player_idx: int, target_sprite_self: Dict[str, Any], spawn_group: str
) -> Dict[str, Any]:
    """Create an avatar object that always sees itself as blue."""
    # Lua is 1-indexed.
    lua_index = player_idx + 1

    # Setup the self vs other sprite mapping.
    source_sprite_self = "Avatar" + str(lua_index)
    custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}

    live_state_name = "player{}".format(lua_index)
    avatar_object = {
        "name": "avatar",
        "components": [
            {
                "component": "StateManager",
                "kwargs": {
                    "initialState": live_state_name,
                    "stateConfigs": [
                        {
                            "state": live_state_name,
                            "layer": "upperPhysical",
                            "sprite": source_sprite_self,
                            "contact": "avatar",
                            "groups": ["players"],
                        },
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
                    "palettes": [shapes.get_palette(colors.human_readable[player_idx])],
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
                    "speed": 1.0,
                    "spawnGroup": "spawnPoints",
                    # "postInitialSpawnGroup": "spawnPoints",
                    "actionOrder": ["move", "turn", "fireZap"],
                    "actionSpec": {
                        "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                        "turn": {"default": 0, "min": -1, "max": 1},
                        "fireZap": {"default": 0, "min": 0, "max": 1},
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
                    "cooldownTime": 2,
                    "beamLength": 2,
                    "beamRadius": 1,
                    "framesTillRespawn": 20,
                    "penaltyForBeingZapped": -10,
                    "rewardForZapping": -1,
                },
            },
            {
                "component": "ReadyToShootObservation",
            },
        ],
    }
    if _ENABLE_DEBUG_OBSERVATIONS:
        avatar_object["components"].append(
            {
                "component": "LocationObserver",
                "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
            }
        )

    return avatar_object


def create_avatar_objects(num_players):
    """Returns list of avatar objects of length 'num_players'."""
    avatar_objects = []
    for player_idx in range(0, num_players):
        spawn_group = "spawnPoints"
        if player_idx < 2:
            # The first two player slots always spawn closer to the apples.
            spawn_group = "insideSpawnPoints"

        game_object = create_avatar_object(
            player_idx, TARGET_SPRITE_SELF, spawn_group=spawn_group
        )
        avatar_objects.append(game_object)

    return avatar_objects


def get_config():
    """Default configuration for training on the commons_harvest level."""
    config = config_dict.ConfigDict()

    # Action set configuration.
    config.action_set = ACTION_SET
    # Observation format configuration.
    config.individual_observation_names = [
        "RGB",
        "READY_TO_SHOOT",
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
            # Debug only (do not use the following observations in policies).
            "WORLD.RGB": specs.rgb(1440, 1920),
        }
    )

    # The roles assigned to each player.
    config.valid_roles = frozenset({"default"})
    config.default_player_roles = ("default",) * 2
    config.lab2d_settings_builder = build

    return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
    """Build substrate definition given player roles."""
    del config
    num_players = len(roles)
    # Build the rest of the substrate definition.
    substrate_definition = dict(
        levelName="commons_harvest",
        levelDirectory="meltingpot/lua/levels",
        numPlayers=num_players,
        # Define upper bound of episode length since episodes end stochastically.
        maxEpisodeLengthFrames=400,
        spriteSize=8,
        topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
        simulation={
            "map": ASCII_MAP,
            "gameObjects": create_avatar_objects(num_players),
            "prefabs": create_prefabs(APPLE_RESPAWN_RADIUS, REGROWTH_PROBABILITIES),
            "charPrefabMap": CHAR_PREFAB_MAP,
            "scene": create_scene(),
        },
    )
    return substrate_definition
