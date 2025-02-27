import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from exploration2d_sb3.models.cnn_encoders import CnnEncoder, CnnMapEncoder


class StackedImgStateExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        device: th.device,
        cnn_encoder_name: str = "CnnEncoder",
        cnn_output_dim: int = 512,
        state_output_dim: int = 32,
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(StackedImgStateExtractor, self).__init__(
            observation_space, features_dim=1
        )

        # Policy Settings

        cnn_class = globals()[cnn_encoder_name]

        # TODO move settings to arguments
        # Image input settings
        self.cfg_image_keys = [
            "local_grid",
            "ego_binary_map",
            "ego_explored_map",
            "ego_goal_map",
            "ego_global_map",
            "global_map",
            "goal_map",
            "pos_map",
            "ego_entropy_map",
            "local_explored_map",
            "local_entropy_map",
        ]
        self.cfg_stacked_image_keys = [
            [
                "ego_binary_map",
                "ego_explored_map",
                "ego_goal_map",
                "ego_global_map",
                "ego_entropy_map",
            ],
            # ["global_map", "goal_map", "pos_map", "binary_map", "explored_map"],
            [
                "local_grid",
                "local_explored_map",
                "local_entropy_map",
            ],
        ]

        # State input settings
        self.cfg_vector_keys = [
            "heading_global_frame",
            "angvel_global_frame",
            "pos_global_frame",
            "vel_global_frame",
            "goal_global_frame",
            "rel_goal",
            "last_goal",
        ]
        self.vector_scales = {
            "heading_global_frame": [-th.pi, th.pi],
            "angvel_global_frame": [-3.0, 3.0],
            "pos_global_frame": [
                th.Tensor([-20.0, -20.0]).to(device),
                th.Tensor([20.0, 20.0]).to(device),
            ],
            "vel_global_frame": [
                th.Tensor([-1.0, -1.0]).to(device),
                th.Tensor([1.0, 1.0]).to(device),
            ],
            "goal_global_frame": [
                th.Tensor([-20.0, -20.0]).to(device),
                th.Tensor([20.0, 20.0]).to(device),
            ],
            "rel_goal": [
                th.Tensor([-20.0, -20.0]).to(device),
                th.Tensor([20.0, 20.0]).to(device),
            ],
            "last_goal": [
                th.Tensor([-1.0, -1.0]).to(device),
                th.Tensor([1.0, 1.0]).to(device),
            ],
        }

        total_concat_size = 0

        self.n_stacks = len(self.cfg_stacked_image_keys)
        n_stacked_img_channels = [0 for _ in range(self.n_stacks)]
        sample_subspaces = [None for _ in range(self.n_stacks)]

        n_states = 0
        self.vector_keys = []
        self.stacked_img_keys = {}
        self.single_img_keys = []

        cnn_encoder_stacks = []
        cnn_encoder_single = []

        for key, subspace in observation_space.spaces.items():
            if key in self.cfg_image_keys:
                # Check if Img belongs to Imgs to be stacked
                stack_idx = np.where(
                    [key in stack_keys for stack_keys in self.cfg_stacked_image_keys]
                )[0]
                stack_idx = stack_idx[0] if stack_idx.size > 0 else None
                # if yes, safe dimensions and for the first obs the obs_space, to create the CNN later
                if stack_idx is not None:
                    if stack_idx not in self.stacked_img_keys:
                        self.stacked_img_keys[stack_idx] = []
                    self.stacked_img_keys[stack_idx].append(key)
                    n_stacked_img_channels[stack_idx] += subspace.shape[0]
                    if sample_subspaces[stack_idx] is None:
                        sample_subspaces[stack_idx] = subspace
                # if not part of a stacked img, create CNN directly
                else:
                    self.single_img_keys.append(key)
                    n_input_channels = subspace.shape[0]
                    cnn_encoder_single.append(
                        cnn_class(
                            n_input_channels=n_input_channels,
                            n_output_features=cnn_output_dim,
                            sample_input=subspace.sample(),
                        )
                    )
                    total_concat_size += cnn_output_dim

            elif key in self.cfg_vector_keys:
                self.vector_keys.append(key)
                n_states += (
                    subspace.shape[0] if len(subspace.shape) == 1 else subspace.shape[1]
                )

        for i in range(self.n_stacks):
            cnn_encoder_stacks.append(None)
            try:
                sample_input = np.stack(
                    [sample_subspaces[i].sample()[0]] * n_stacked_img_channels[i]
                )
            except ():
                raise ValueError(
                    "Image observations to be stacked must have the same shape!"
                )
            cnn_encoder_stacks[i] = cnn_class(
                n_input_channels=n_stacked_img_channels[i],
                n_output_features=cnn_output_dim,
                sample_input=sample_input,
            )
            total_concat_size += cnn_output_dim

        self.cnn_encoder_stacks = nn.ModuleList(cnn_encoder_stacks)
        self.cnn_encoder_single = nn.ModuleList(cnn_encoder_single)

        # self.state_encoder = nn.Flatten()
        # total_concat_size += n_states  # get_flattened_obs_dim(subspace)

        self.state_encoder = nn.Linear(
            in_features=n_states, out_features=state_output_dim
        )
        total_concat_size += state_output_dim

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # Process Stacked Images
        for i, cnn in enumerate(self.cnn_encoder_stacks):
            images_list = [observations[key] for key in self.stacked_img_keys[i]]
            stacked_images = th.cat(images_list, dim=1).float() / 255.0
            encoded_tensor_list.append(self.cnn_encoder_stacks[i](stacked_images))

        # Process Single images
        for i, cnn in enumerate(self.cnn_encoder_single):
            image = observations[self.single_img_keys[i]].float() / 255.0
            encoded_tensor_list.append(self.cnn_encoder_single[i](image))

        # Process States
        states_list = []
        for key in self.vector_keys:
            states_list.append(
                (
                    (observations[key] - self.vector_scales[key][0])
                    * 2
                    / (self.vector_scales[key][1] - self.vector_scales[key][0])
                )
                - 1.0
            )
        states = th.cat(states_list, dim=1)
        encoded_tensor_list.append(self.state_encoder(states))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
