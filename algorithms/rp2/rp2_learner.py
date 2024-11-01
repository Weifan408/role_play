import logging
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

import torch.nn.functional as F
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ModuleID, TensorType

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

ENCODER_DECODER_KEY = "encoder_decoder_loss"
PREDICTED_OTHER_THETA_LOSS_KEY = "predicted_theta_loss"


class RPTorchLearner(PPOTorchLearner):

    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: Optional["AlgorithmConfig"] = None,
        batch: NestedDict,
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        total_loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        if "predicted_out" not in fwd_out:
            predicted_other_theta_loss = 0
        else:
            criterion = nn.CrossEntropyLoss()
            other_theta = batch[Columns.OBS]["other_theta"]  # shape is [2000, 20, 7]
            predicted_out = fwd_out["predicted_out"]  # shape is [2000, 20, 7, 3]
            predicted_out = predicted_out.reshape(-1, predicted_out.shape[-1])
            other_theta = other_theta.reshape(-1)
            predicted_other_theta_loss = criterion(predicted_out, other_theta)

        total_loss += predicted_other_theta_loss
        self.register_metrics(
            module_id,
            {
                PREDICTED_OTHER_THETA_LOSS_KEY: predicted_other_theta_loss,
            },
        )
        return total_loss
