import torch
import torch.nn as nn
from stable_baselines3.td3.policies import TD3Policy

class PlannerNetwork(nn.Module):
    """Input: obj_pos + cur_state; Output: next_planning_state"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class CustomPlannerTD3Policy(TD3Policy):

    def _build_actor(self) -> None:
        input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.shape[0]
        self.actor = PlannerNetwork(input_dim, output_dim)
        self.actor_target = PlannerNetwork(input_dim, output_dim)
