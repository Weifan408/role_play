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

        self.multiplayers = catalog.multiplayers
        if catalog.multiplayers:
            self.other_theta_encoder = catalog.other_theta_encoder
            self.other_theta_decoder = catalog.other_theta_decoder
        else:
            self.other_theta_encoder = None
            self.other_theta_decoder = None
        
        self.discriminator = catalog.discriminator
        self.encoder = catalog.encoder
        self.pi_head = catalog.pi_head
        self.vf_head = catalog.vf_head

        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)

    @override(PPORLModule)
    def get_initial_state(self) -> dict:
        return self.encoder.get_initial_state()

    def get_encoder_decoder_out(self, other_theta) -> Mapping[str, Any]:
        other_theta_encoder_out = self.other_theta_encoder(other_theta)
        other_theta_decoder_out = self.other_theta_decoder(other_theta_encoder_out)
        return {"other_theta_encoder_out": other_theta_encoder_out, "other_theta_decoder_out": other_theta_decoder_out}

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        encoder_outs = self.encoder(batch)
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        
        theta = batch[Columns.OBS]['theta']
        other_theta = batch[Columns.OBS]['other_theta']        
        discriminator_input = torch.cat([encoder_outs[ENCODER_OUT][ACTOR], theta], dim=-1)
        predicted_out = self.discriminator(discriminator_input)

        if self.multiplayers:
            with torch.no_grad():
                predicted_out = self.other_theta_decoder(predicted_out)
            output.update(self.get_encoder_decoder_out(other_theta))
        output["predicted_out"] = predicted_out
        pi_head_input = torch.cat([encoder_outs[ENCODER_OUT][ACTOR], theta, predicted_out], dim=-1)

        vf_head_input = torch.cat([encoder_outs[ENCODER_OUT][CRITIC], theta, other_theta], dim=-1)
        vf_out = self.vf_head(vf_head_input)
        output[Columns.VF_PREDS] = vf_out.squeeze(-1)

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

        encoder_outs = self.encoder(batch)
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        
        theta = batch[Columns.OBS]['theta']
        other_theta = batch[Columns.OBS]['other_theta']        
        discriminator_input = torch.cat([encoder_outs[ENCODER_OUT][ACTOR], theta], dim=-1)
        predicted_out = self.discriminator(discriminator_input)

        if self.multiplayers:
            with torch.no_grad():
                predicted_out = self.other_theta_decoder(predicted_out)
            output.update(self.get_encoder_decoder_out(other_theta))
        output["predicted_out"] = predicted_out
        pi_head_input = torch.cat([encoder_outs[ENCODER_OUT][ACTOR], theta, predicted_out], dim=-1)

        vf_head_input = torch.cat([encoder_outs[ENCODER_OUT][CRITIC], theta, other_theta], dim=-1)
        vf_out = self.vf_head(vf_head_input)
        output[Columns.VF_PREDS] = vf_out.squeeze(-1)

        action_logits = self.pi_head(pi_head_input)
        output[Columns.ACTION_DIST_INPUTS] = action_logits

        return output
        
    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        encoder_outs = self.encoder(batch)
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        
        theta = batch[Columns.OBS]['theta']
        other_theta = batch[Columns.OBS]['other_theta']        
        discriminator_input = torch.cat([encoder_outs[ENCODER_OUT][ACTOR], theta], dim=-1)
        predicted_out = self.discriminator(discriminator_input)

        if self.multiplayers:
            with torch.no_grad():
                predicted_out = self.other_theta_decoder(predicted_out)
            output.update(self.get_encoder_decoder_out(other_theta))
        output["predicted_out"] = predicted_out
        pi_head_input = torch.cat([encoder_outs[ENCODER_OUT][ACTOR], theta, predicted_out], dim=-1)
        vf_head_input = torch.cat([encoder_outs[ENCODER_OUT][CRITIC], theta, other_theta], dim=-1)
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

