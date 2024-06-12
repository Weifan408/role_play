import gymnasium as gym
import numpy as np
from numpy.random import rand
from gymnasium.spaces import Box, Dict, Discrete

from envs.social_dilemmas.agent import CleanupAgent
from envs.social_dilemmas.map_env import MapEnv
from envs.social_dilemmas.maps import CLEANUP_MAP

# Add custom actions to the agent
_CLEANUP_ACTIONS = {"FIRE": 3, "CLEAN": 3}  # length of firing beam, length of cleanup beam

# Custom colour dictionary
CLEANUP_COLORS = {
    b"C": np.array([100, 255, 255], dtype=np.uint8),  # Cyan cleaning beam
    b"S": np.array([113, 75, 24], dtype=np.uint8),  # Light grey-blue stream cell
    b"H": np.array([99, 156, 194], dtype=np.uint8),  # Brown waste cells
    b"R": np.array([113, 75, 24], dtype=np.uint8),  # Light grey-blue river cell
}

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

CLEANUP_VIEW_SIZE = 7

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.1


class CustomCleanupEnv(MapEnv):
    def __init__(
        self,
        ascii_map=CLEANUP_MAP,
        num_agents=2,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        view_size = CLEANUP_VIEW_SIZE,
        alpha=0.0,
        beta=0.0,
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            view_size,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()
        self.players_ids = list(self.agents.keys())

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)

    @property
    def action_space(self):
        return gym.spaces.Discrete(9)

    def step(self, actions):
        self.beam_pos = []
        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        # Remove agents from color map
        for agent in self.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            self.single_update_world_color_map(row, col, self.world_map[row, col])

        self.update_moves(agent_actions)

        for agent in self.agents.values():
            pos = agent.pos
            if agent.removed:
                agent.has_removed_steps += 1
                if agent.has_removed_steps >= agent.removed_steps:
                    agent.reborn(self.spawn_point())
            else:
                new_char = agent.consume(self.world_map[pos[0], pos[1]])
                self.single_update_map(pos[0], pos[1], new_char)

        # execute custom moves like firing
        self.update_custom_moves(agent_actions)

        # execute spawning events
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()
        # Add agents to color map
        for agent in self.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            # Firing beams have priority over agents and should cover them
            if self.world_map[row, col] not in [b"F", b"C"]:
                self.single_update_world_color_map(row, col, agent.get_char_id())

        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = self.color_view(agent)
            if len(rgb_arr) == 0:
                rgb_arr = np.zeros(self.observation_space['curr_obs'].shape, dtype=np.uint8)
            ready_to_shoot = 1 if agent.removed else 0
            # concatenate on the prev_actions to the observations
            if self.return_agent_actions:
                prev_actions = np.array(
                    [actions[key] for key in sorted(actions.keys()) if key != agent.agent_id]
                ).astype(np.uint8)
                visible_agents = self.find_visible_agents(agent.agent_id)
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "ready_to_shoot": ready_to_shoot,
                    "other_agent_actions": prev_actions,
                    "visible_agents": visible_agents,
                    "prev_visible_agents": agent.prev_visible_agents,
                }
                agent.prev_visible_agents = visible_agents
            else:
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "ready_to_shoot": ready_to_shoot,
                }
            rewards[agent.agent_id] = agent.compute_reward()
            dones[agent.agent_id] = agent.get_done()
            infos[agent.agent_id] = {}
        
            self.total_rewards[agent.agent_id] += rewards[agent.agent_id]

        if self.use_collective_reward:
            collective_reward = sum(rewards.values())
            for agent in rewards.keys():
                rewards[agent] = collective_reward
        if self.inequity_averse_reward:
            assert self.num_agents > 1, "Cannot use inequity aversion with only one agent!"
            temp_rewards = rewards.copy()
            for agent in rewards.keys():
                diff = np.array([r - rewards[agent] for r in rewards.values()])
                dis_inequity = self.alpha * sum(diff[diff > 0])
                adv_inequity = self.beta * sum(diff[diff < 0])
                temp_rewards[agent] -= (dis_inequity + adv_inequity) / (self.num_agents - 1)
            rewards = temp_rewards
        
        dones["__all__"] = np.any(list(dones.values()))
        return observations, rewards, dones, dones, infos

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        return updates

    def reset(self):
        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.custom_map_update()
        self.total_rewards = {agent_id: 0 for agent_id in self.agents}
        map_with_agents = self.get_map_with_agents()

        observations = {}
        for agent in self.agents.values():
            agent.full_map = map_with_agents
            rgb_arr = self.color_view(agent)
            # concatenate on the prev_actions to the observations
            if self.return_agent_actions:
                # No previous actions so just pass in "wait" action
                prev_actions = np.array([4 for _ in range(self.num_agents - 1)]).astype(np.uint8)
                visible_agents = self.find_visible_agents(agent.agent_id)
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "ready_to_shoot": 1,
                    "other_agent_actions": prev_actions,
                    "visible_agents": visible_agents,
                    "prev_visible_agents": visible_agents,
                }
                agent.prev_visible_agents = visible_agents
            else:
                observations[agent.agent_id] = {
                    "curr_obs": rgb_arr,
                    "ready_to_shoot": 1,
                }
        return observations, {}

    def custom_map_update(self):
        """ "Update the probabilities and then spawn"""
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id,
                spawn_point,
                rotation,
                map_with_agents,
                view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            np.random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (
                    1
                    - (waste_density - thresholdRestoration)
                    / (thresholdDepletion - thresholdRestoration)
                ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area

    @property
    def observation_space(self):
        obs_space = {
            "curr_obs": Box(
                low=0,
                high=255,
                shape=(self.view_len * 2 + 1, self.view_len * 2 + 1, 3),
                dtype=np.uint8,
            ),
            "ready_to_shoot": Discrete(2),
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