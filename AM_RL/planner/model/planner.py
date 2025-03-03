import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy

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


class CustomDQNPolicy(DQNPolicy):
    
    def _build_q_net(self):
        input_dim = 22
        output_dim = 19
        self.q_net = PlannerNetwork(input_dim, output_dim)
        self.q_net_target = PlannerNetwork(input_dim, output_dim)
    
    def forward(self, obs, deterministic: bool = True):
        q_values = self.q_net(obs)
        if deterministic:
            return torch.argmax(q_values, dim=1)
        else:
            return q_values
