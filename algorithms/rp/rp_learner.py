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
PREDICTED_THETA_LOSS_KEY = "predicted_theta_loss"


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
        if "loss_mask" in batch:
            num_valid = torch.sum(batch["loss_mask"])

            def possibly_masked_mean(data_):
                return torch.sum(data_[batch["loss_mask"]]) / num_valid

        else:
            possibly_masked_mean = torch.mean
        
        criterion = nn.BCELoss()
        if "other_theta_decoder_out" in fwd_out:
            # encoder-decoder loss
            other_theta_decoder_out = fwd_out["other_theta_decoder_out"]
            other_theta_decoder_out_softmax = F.softmax(other_theta_decoder_out, dim=-1)
            encoder_decoder_loss = possibly_masked_mean(criterion(other_theta_decoder_out_softmax, batch[Columns.OBS]['other_theta']))

            total_loss += encoder_decoder_loss
            self.register_metrics(
                module_id,
                {
                    ENCODER_DECODER_KEY: encoder_decoder_loss,
                },
            )

        predicted_out = fwd_out["predicted_out"]
        predicted_out_softmax = F.softmax(predicted_out, dim=-1)
        predicted_theta_loss = possibly_masked_mean(criterion(predicted_out_softmax, batch[Columns.OBS]['other_theta']))
        total_loss += predicted_theta_loss

        self.register_metrics(
            module_id,
            {
                PREDICTED_THETA_LOSS_KEY: predicted_theta_loss,
            },
        )
        return total_loss