import torch
import torch.nn as nn
from stable_baselines3.td3.policies import TD3Policy

class PlannerNetwork(nn.Module):
    """Input: cur_state + obj_pos; Output: next_planning_state"""

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

    def _build_actor(self, lr_schedule) -> None:
        input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.shape[0]
        self.actor = PlannerNetwork(input_dim, output_dim)
        self.actor_target = PlannerNetwork(input_dim, output_dim)

        pretraining_path = "pretraining_planner.pth"
        self.actor.load_state_dict(torch.load(pretraining_path, map_location=self.device))
        self.actor_target.load_state_dict(torch.load(pretraining_path, map_location=self.device))

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()

        self.actor_optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.critic_optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)