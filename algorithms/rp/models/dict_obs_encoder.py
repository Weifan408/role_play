import gymnasium as gym

from ray.rllib.core.models.torch.primitives import TorchCNN, TorchMLP
from ray.rllib.utils.framework import try_import_torch
from algorithms.rp.utils import get_cnn_output_dims 

torch, nn = try_import_torch()


class DictObsEncoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
    ):
        super().__init__()
        self.output_dims = []
        if 'cnn' in observation_space.spaces:
            self.cnn_encoder = TorchCNN(
                input_dims=observation_space['cnn'].shape,
                cnn_filter_specifiers=model_config_dict["conv_filters"],
                cnn_use_bias=True,
                cnn_use_layernorm=True,
            )
            cnn_encoder_output_dims = get_cnn_output_dims(observation_space['cnn'].shape,
                                                          model_config_dict["conv_filters"])
            self.output_dims.append(cnn_encoder_output_dims)

        if 'mlp' in observation_space.spaces:
            if model_config_dict["encoder_latent_dim"]:
                hidden_layer_dims = model_config_dict["fcnet_hiddens"]
                encoder_latent_dim = model_config_dict["encoder_latent_dim"]
            else:
                hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
                encoder_latent_dim = model_config_dict["fcnet_hiddens"][-1]
            self.mlp_encoder = TorchMLP(
                input_dim=observation_space['mlp'].shape[0],
                hidden_layer_dims=hidden_layer_dims,
                hidden_layer_use_layernorm= True,
                output_dim=encoder_latent_dim,
                hidden_layer_activation=model_config_dict["fcnet_activation"],
            )
            self.output_dims.append(encoder_latent_dim)
        self.output_dim = sum(self.output_dims)
    
    def forward(self, inputs):
        outputs = []

        if 'cnn' in inputs['obs']:
            # [B, T, H, W, C] -> [BxT, H, W, C]
            cnn_input = inputs['obs']['cnn'].reshape(-1, *inputs['obs']['cnn'].shape[2:])
            cnn_encoder_out = self.cnn_encoder(cnn_input)
            # [BxT, H, W, C] -> [B, T, D]
            cnn_encoder_out = cnn_encoder_out.reshape(*inputs['obs']['cnn'].shape[:2], -1)
            outputs.append(cnn_encoder_out)

        if 'mlp' in inputs['obs']:
            mlp_encoder_out = self.mlp_encoder(inputs['obs']['mlp'])
            outputs.append(mlp_encoder_out)

        if outputs:
            return torch.cat(outputs, dim=1)
        else:
            raise ValueError("No valid observation space found in input dict.")
