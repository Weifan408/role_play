import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from numpy.random import rand

from envs.social_dilemmas.agent import HarvestAgent
from envs.social_dilemmas.map_env import MapEnv
from envs.social_dilemmas.maps import HARVEST_MAP

APPLE_RADIUS = 2

# Add custom actions to the agent
_HARVEST_ACTIONS = {"FIRE": 5}  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

HARVEST_VIEW_SIZE = 3


class HarvestEnv(MapEnv):
    def __init__(
        self,
        ascii_map=HARVEST_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
        flatten_rgb_obs = True,
        world_obs = False,
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
            flatten_rgb_obs=flatten_rgb_obs,
        )
        self.world_obs = world_obs
        self.players_ids = list(self.agents.keys())
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])
        

    @property
    def action_space(self):
        return gym.spaces.Discrete(8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

    def custom_action(self, agent, action):
        agent.fire_beam(b"F")
        updates = self.update_map_fire(
            agent.pos.tolist(),
            agent.get_orientation(),
            self.all_actions["FIRE"],
            fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                0 <= x + j < self.world_map.shape[0]
                                and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples

    def color_view(self, agent):
        row, col = agent.pos[0], agent.pos[1]

        if self.world_obs:
            return self.world_map_color[self.map_padding: -self.map_padding,
                self.map_padding:-self.map_padding,
            ]
        else:
            view_slice = self.world_map_color[
                row + self.map_padding - self.view_len : row + self.map_padding + self.view_len + 1,
                col + self.map_padding - self.view_len : col + self.map_padding + self.view_len + 1,
            ]
            if agent.orientation == "UP":
                rotated_view = view_slice
            elif agent.orientation == "LEFT":
                rotated_view = np.rot90(view_slice)
            elif agent.orientation == "DOWN":
                rotated_view = np.rot90(view_slice, k=2)
            elif agent.orientation == "RIGHT":
                rotated_view = np.rot90(view_slice, k=1, axes=(1, 0))
            return rotated_view
    
    @property
    def observation_space(self):
        obs_space = {
            "curr_obs": Box(
                low=0,
                high=255,
                shape=(self.world_map.shape[0], self.world_map.shape[1], 3),
                dtype=np.uint8,
            ) if self.world_obs else Box(
                low=0,
                high=255,
                shape=(self.view_len * 2 + 1, self.view_len * 2 + 1, 3),
                dtype=np.uint8,
            )
        }

        if self.return_agent_actions:
            # Append the actions of other agents
            obs_space = {
                **obs_space,
                "other_agent_actions": Box(
                    low=0,
                    high=len(self.all_actions),
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "visible_agents": Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
                "prev_visible_agents": Box(
                    low=0,
                    high=1,
                    shape=(self.num_agents - 1,),
                    dtype=np.uint8,
                ),
            }
        obs_space = Dict(obs_space)
        obs_space.dtype = np.uint8
        return obs_space