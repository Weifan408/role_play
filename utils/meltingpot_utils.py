# Copyright 2020 DeepMind Technologies Limited.
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
"""MeltingPotEnv as a MultiAgentEnv wrapper to interface with RLLib."""

from typing import Tuple

import dm_env


from meltingpot import substrate
from meltingpot.utils.policies import policy

from ray.rllib import algorithms
from ray.rllib.policy import sample_batch





class RayModelPolicy(policy.Policy[policy.State]):
	"""Policy wrapping an RLLib model for inference.

	Note: Currently only supports a single input, batching is not enabled
	"""

	def __init__(self,
				model: algorithms.Algorithm,
				policy_id: str = sample_batch.DEFAULT_POLICY_ID) -> None:
		"""Initialize a policy instance.

		Args:
		model: An rllib.trainer.Trainer checkpoint.
		policy_id: Which policy to use (if trained in multi_agent mode)
		"""
		self._model = model
		self._prev_action = 0
		self._policy_id = policy_id

	def step(self, timestep: dm_env.TimeStep,
			prev_state: policy.State) -> Tuple[int, policy.State]:
		"""See base class."""
		observations = {
			key: value
			for key, value in timestep.observation.items()
			if 'WORLD' not in key
		}

		action, state, _ = self._model.compute_single_action(
			observations,
			prev_state,
			policy_id=self._policy_id,
			prev_action=self._prev_action,
			prev_reward=timestep.reward)

		self._prev_action = action
		return action, state

	def initial_state(self) -> policy.State:
		"""See base class."""
		self._prev_action = 0
		return self._model.get_policy(self._policy_id).get_initial_state()

	def close(self) -> None:
		"""See base class."""