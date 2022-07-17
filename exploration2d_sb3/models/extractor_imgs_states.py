import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class ImgStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(ImgStateExtractor, self).__init__(observation_space, features_dim=1)

        # Policy Settings
        cnn_output_dim = 256
        self.image_keys = ["local_grid", "ego_binary_map", "mc_ego_binary_goal"]
        self.vector_keys = [
            "heading_global_frame",
            "angvel_global_frame",
            "pos_global_frame",
            "vel_global_frame",
        ]
        self.vector_scales = {
            "heading_global_frame": [-th.pi, th.pi],
            "angvel_global_frame": [-3.0, 3.0],
            "pos_global_frame": [th.Tensor([-10.0, -10.0]), th.Tensor([10.0, 10.0])],
            "vel_global_frame": [th.Tensor([-3.0, -3.0]), th.Tensor([3.0, 3.0])],
        }

        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key in self.image_keys:
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            elif key in self.vector_keys:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():

            # Preprocess obs
            if key in self.image_keys:
                obs = observations[key].float() / 255.0
            elif key in self.vector_keys:
                obs = (
                    (observations[key] - self.vector_scales[key][0])
                    * 2
                    / (self.vector_scales[key][1] - self.vector_scales[key][0])
                ) - 1.0
            else:
                obs = observations[key]

            encoded_tensor_list.append(extractor(obs))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
