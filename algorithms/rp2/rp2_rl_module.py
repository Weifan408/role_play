from typing import Mapping, Any

import numpy as np

from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

torch, nn = try_import_torch()


class RPTorchRLModule(TorchRLModule, PPORLModule):
    framework: str = "torch"

    def setup(self):
        catalog = self.config.get_catalog()

        self.predictors = catalog.predictors
        self.actor_encoder = catalog.actor_encoder
        self.critic_encoder = catalog.critic_encoder

        self.pi_head = catalog.pi_head
        self.vf_head = catalog.vf_head
        self.class_num = catalog.class_num
        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)
        self.train_cnt = 0

    @override(PPORLModule)
    def get_initial_state(self) -> dict:
        initial_state = {}
        if self.actor_encoder.use_rnn:
            initial_state[ACTOR] = self.actor_encoder.get_initial_state()
        if self.critic_encoder.use_rnn:
            initial_state[CRITIC] = self.critic_encoder.get_initial_state()
        return initial_state

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        actor_encoder_out, actor_h_next = self.actor_encoder(batch)

        output[Columns.STATE_OUT] = {
            ACTOR: {"h": actor_h_next},
        }

        theta = batch[Columns.OBS]["theta"]
        other_theta = batch[Columns.OBS]["other_theta"]

        theta_one_hot = nn.functional.one_hot(theta, num_classes=self.class_num)
        theta_one_hot = theta_one_hot.reshape(*theta_one_hot.shape[:-2], -1)
        other_theta_one_hot = nn.functional.one_hot(
            other_theta, num_classes=self.class_num
        )
        other_theta_one_hot = other_theta_one_hot.reshape(
            *other_theta_one_hot.shape[:-2], -1
        )

        predicted_outs = []
        for discriminator in self.discriminators:
            predicted_outs.append(discriminator(actor_encoder_out))

        predicted_outs = torch.stack(predicted_outs)
        predicted_outs = predicted_outs.permute(1, 2, 0, 3)
        output["predicted_out"] = predicted_outs
        differentiable_predicted_sample = torch.nn.functional.gumbel_softmax(
            predicted_outs, tau=0.1, hard=True
        )
        differentiable_predicted_sample = differentiable_predicted_sample.reshape(
            *differentiable_predicted_sample.shape[:-2], -1
        )
        output["predicted_out_one_hot_sample"] = differentiable_predicted_sample
        pi_head_input = torch.cat(
            [actor_encoder_out, theta_one_hot, differentiable_predicted_sample],
            dim=-1,
        )

        action_logits = self.pi_head(pi_head_input)
        output[Columns.ACTION_DIST_INPUTS] = action_logits

        return output

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        """PPO forward pass during exploration.
        Besides the action distribution, this method also returns the parameters of the
        policy distribution to be used for computing KL divergence between the old
        policy and the new policy during training.
        """
        output = {}

        actor_encoder_out, actor_h_next = self.actor_encoder(batch)
        critic_encoder_out, critic_h_next = self.critic_encoder(batch)

        output[Columns.STATE_OUT] = {
            ACTOR: {"h": actor_h_next},
        }

        theta = batch[Columns.OBS]["theta"]
        other_theta = batch[Columns.OBS]["other_theta"]

        theta_one_hot = nn.functional.one_hot(theta, num_classes=self.class_num)
        theta_one_hot = theta_one_hot.reshape(*theta_one_hot.shape[:-2], -1)
        other_theta_one_hot = nn.functional.one_hot(
            other_theta, num_classes=self.class_num
        )
        other_theta_one_hot = other_theta_one_hot.reshape(
            *other_theta_one_hot.shape[:-2], -1
        )

        if self.train_cnt < 60000:
            pi_head_input = torch.cat(
                [actor_encoder_out, theta_one_hot, other_theta_one_hot], dim=-1
            )
        else:
            predicted_outs = []
            for discriminator in self.discriminators:
                predicted_outs.append(discriminator(actor_encoder_out))

            predicted_outs = torch.stack(predicted_outs)
            predicted_outs = predicted_outs.permute(1, 2, 0, 3)
            output["predicted_out"] = predicted_outs

            differentiable_predicted_sample = torch.nn.functional.gumbel_softmax(
                predicted_outs, tau=0.1, hard=True
            )
            differentiable_predicted_sample = differentiable_predicted_sample.reshape(
                *differentiable_predicted_sample.shape[:-2], -1
            )
            output["predicted_out_one_hot_sample"] = differentiable_predicted_sample
            pi_head_input = torch.cat(
                [actor_encoder_out, theta_one_hot, differentiable_predicted_sample],
                dim=-1,
            )

        vf_head_input = torch.cat(
            [critic_encoder_out, theta_one_hot, other_theta_one_hot], dim=-1
        )
        vf_out = self.vf_head(vf_head_input)
        output[Columns.VF_PREDS] = vf_out.squeeze(-1)

        action_logits = self.pi_head(pi_head_input)
        output[Columns.ACTION_DIST_INPUTS] = action_logits

        return output

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}
        self.train_cnt += 1
        actor_encoder_out, actor_h_next = self.actor_encoder(batch)
        critic_encoder_out, critic_h_next = self.critic_encoder(batch)

        output[Columns.STATE_OUT] = {
            ACTOR: {"h": actor_h_next},
        }

        theta = batch[Columns.OBS]["theta"]
        other_theta = batch[Columns.OBS]["other_theta"]
        theta_one_hot = nn.functional.one_hot(theta, num_classes=self.class_num)
        theta_one_hot = theta_one_hot.reshape(*theta_one_hot.shape[:-2], -1)
        other_theta_one_hot = nn.functional.one_hot(
            other_theta, num_classes=self.class_num
        )
        other_theta_one_hot = other_theta_one_hot.reshape(
            *other_theta_one_hot.shape[:-2], -1
        )

        if self.train_cnt < 60000:
            if self.train_cnt > 40000:
                predicted_outs = []
                for discriminator in self.discriminators:
                    predicted_outs.append(discriminator(actor_encoder_out))
                predicted_outs = torch.stack(predicted_outs)
                predicted_outs = predicted_outs.permute(1, 2, 0, 3)
                output["predicted_out"] = predicted_outs
            pi_head_input = torch.cat(
                [actor_encoder_out, theta_one_hot, other_theta_one_hot], dim=-1
            )
        else:
            predicted_outs = []
            for discriminator in self.discriminators:
                predicted_outs.append(discriminator(actor_encoder_out))
            predicted_outs = torch.stack(predicted_outs)
            predicted_outs = predicted_outs.permute(1, 2, 0, 3)
            output["predicted_out"] = predicted_outs
            differentiable_predicted_sample = torch.nn.functional.gumbel_softmax(
                predicted_outs, tau=0.1, hard=True
            )
            differentiable_predicted_sample = differentiable_predicted_sample.reshape(
                *differentiable_predicted_sample.shape[:-2], -1
            )
            output["predicted_out_one_hot_sample"] = differentiable_predicted_sample
            pi_head_input = torch.cat(
                [actor_encoder_out, theta_one_hot, differentiable_predicted_sample],
                dim=-1,
            )

        vf_head_input = torch.cat(
            [critic_encoder_out, theta_one_hot, other_theta_one_hot], dim=-1
        )
        vf_out = self.vf_head(vf_head_input)
        output[Columns.VF_PREDS] = vf_out.squeeze(-1)

        action_logits = self.pi_head(pi_head_input)
        output[Columns.ACTION_DIST_INPUTS] = action_logits

        return output

    @override(PPORLModule)
    def _compute_values(self, batch, device=None):
        infos = batch.pop(Columns.INFOS, None)
        batch = convert_to_torch_tensor(batch, device=device)
        if infos is not None:
            batch[Columns.INFOS] = infos

        return self.critic(batch)
