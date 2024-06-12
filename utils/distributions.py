import gymnasium as gym
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import tree
import abc


from ray.rllib.models.distributions import Distribution
from ray.rllib.models.torch.torch_distributions import TorchDistribution, TorchDeterministic
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, Union, Tuple

from torchrl.modules.distributions import TruncatedNormal
from torch.distributions import Normal

torch, nn = try_import_torch()


class TorchTruncatedGaussian(TorchDistribution):
    LOW = -np.pi
    UP = np.pi

    @override(TorchDistribution)
    def __init__(
        self,
        loc: Union[float, torch.Tensor],
        scale: Optional[Union[float, torch.Tensor]],
        # upscale: Optional[Union[float, torch.Tensor]],
    ):
        self.loc = loc
        self.scale = scale
        self.alpha = (self.LOW - loc) / scale
        self.beta = (self.UP - loc) / scale
        self.standard_normal_dist = Normal(0, 1)
        self.z = self.standard_normal_dist.cdf(self.beta) - self.standard_normal_dist.cdf(self.alpha)
        super().__init__(loc=loc, scale=scale)

    def _get_torch_distribution(self, loc, scale) -> "torch.distributions.Distribution":
        return TruncatedNormal(loc=loc, scale=scale, min=self.LOW, max=self.UP)

    @override(TorchDistribution)
    def logp(self, value: TensorType) -> TensorType:
        return super().logp(value)

    @override(TorchDistribution)
    def entropy(self) -> TensorType:
        # /phi(/alpha) /phi(/beta)
        normal_pdf_min = torch.exp(self.standard_normal_dist.log_prob(self.alpha))
        normal_pdf_max = torch.exp(self.standard_normal_dist.log_prob(self.beta))

        entropy =  torch.log(self.z * self.scale * torch.sqrt(2 * torch.pi * torch.exp(torch.tensor(1.)))) + (self.alpha * normal_pdf_min - self.beta * normal_pdf_max) / (2 * self.z)
        
        return entropy.sum(-1)

    @override(TorchDistribution)
    def kl(self, other: "TruncatedNormal") -> TensorType:
        # using monte carlo method to calculate kl divergence
        samples = self.sample(sample_shape=[10000])
        log_pdf_ratio = self.logp(samples) - other.logp(samples)
        return log_pdf_ratio.mean(0)

    @staticmethod
    @override(Distribution)
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        assert isinstance(space, gym.spaces.Box)
        return int(np.prod(space.shape, dtype=np.int32) * 2)

    @classmethod
    @override(Distribution)
    def from_logits(cls, logits: TensorType, **kwargs) -> "TorchTruncatedGaussian":
        loc, log_std = logits.chunk(2, dim=-1)
        scale = log_std.exp()
        return TorchTruncatedGaussian(loc=loc, scale=scale)

    def to_deterministic(self) -> "TorchDeterministic":
        return TorchDeterministic(loc=self.loc)