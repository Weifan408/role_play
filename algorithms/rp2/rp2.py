from typing import Optional, Type, Union, TYPE_CHECKING

from ray.rllib import Policy
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo import PPOConfig, PPO
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import override

from algorithms.rp2.rp2_catalog import RPCatalog

if TYPE_CHECKING:
    from ray.rllib.core.learner.learner import Learner


class RPConfig(PPOConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class=algo_class or RP)

    @override(PPOConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:
        if self.framework_str == "torch":
            from algorithms.rp2.rp2_rl_module import RPTorchRLModule

            return SingleAgentRLModuleSpec(
                module_class=RPTorchRLModule, catalog_class=RPCatalog
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use either 'torch'."
            )

    @override(PPOConfig)
    def get_default_learner_class(self) -> Union[Type["Learner"], str]:
        if self.framework_str == "torch":
            from algorithms.rp2.rp2_learner import RPTorchLearner

            return RPTorchLearner
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use either 'torch' or 'tf2'."
            )


class RP(PPO):
    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return RPConfig()
