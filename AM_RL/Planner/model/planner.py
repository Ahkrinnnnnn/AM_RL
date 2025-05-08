import os

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.td3.policies import Actor, TD3Policy
import AM_RL

class PlannerNetwork(Actor):
    """Input: cur_state + obj_pos; Output: next_planning_state"""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=[256, 256],
            features_extractor=None,
            features_dim=observation_space.shape[0],
            activation_fn=nn.Tanh,
            normalize_images=False,
        )
        self.mu = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_space.shape[0]),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mu(x)


class CustomPlannerTD3Policy(TD3Policy):

    def _build(self, lr_schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = PlannerNetwork(self.observation_space, self.action_space)
        self.actor_target = PlannerNetwork(self.observation_space, self.action_space)
        # Initialize the target to have the same weights as the actor
        pretraining_path = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/Planner/model/pretraining_planner.pth"
        self.actor.load_state_dict(torch.load(pretraining_path, map_location=self.device, weights_only=True))
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)