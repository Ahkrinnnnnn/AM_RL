import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class ControllerNetwork(nn.Module):
    """Input: cur_state + obj_pos + next_planning_state; Output: thrust + torque"""

    def __init__(self, fearture_dim, last_layer_dim_pi, last_layer_dim_vf):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(
            nn.Linear(fearture_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, last_layer_dim_pi)
        )

        self.value_net = nn.Sequential(
            nn.Linear(fearture_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, last_layer_dim_vf),
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class CustomControllerAC(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, custom_param, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, custom_param, *args, **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ControllerNetwork(self.features_dim, 256, 1)
        