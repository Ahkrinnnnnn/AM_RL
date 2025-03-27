import os

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.ppo.policies import MlpPolicy
import AM_RL

class ControllerNetwork(MlpExtractor):
    """Input: cur_state + obj_pos + next_planning_state; Output: thrust + torque"""

    def __init__(self, pretraining: bool = False):
        super().__init__(
            feature_dim = 41,
            net_arch = [256, 256],
            activation_fn = nn.Tanh,
            device = "auto"
        )
        self.output = nn.Sequential(
            nn.Linear(self.latent_dim_pi, 9),
            nn.Tanh()
        )
        self.pretraining = pretraining

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        if self.pretraining:
            return self.output(self.policy_net(features))
        return self.policy_net(features)


class CustomControllerPPOPolicy(MlpPolicy):

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = ControllerNetwork

        pretraining_path = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/Controller/model/pretarining_controller.pth"
        pretrained_dict = torch.load(pretraining_path, map_location=self.device)
        filtered_dict = {k: v for k, v in pretrained_dict.items() if 'policy_net' in k}
        self.mlp_extractor.policy_net.load_state_dict(filtered_dict, strict=False)
