import os

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, mdp
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import AM_RL
from AM_RL.Planner.cfgs.robotCfg import *
from AM_RL.Planner.cfgs.rewardCfg import *

norm_path = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/Planner/model/pnorm_params.npz"
norm_params = np.load(norm_path)

states_mid = torch.tensor(norm_params["states_mid"], device="cuda")
states_range = torch.tensor(norm_params["states_range"], device="cuda")
action_mid = torch.tensor(norm_params["action_mid"], device="cuda")
action_range = torch.tensor(norm_params["action_range"], device="cuda")


##
# Customize functions
##


def robot_out_of_bounds(env: ManagerBasedRLEnv, asset_name: str, bounds: list):
    root_pos_w = env.scene[asset_name].data.root_pos_w
    condition = (
        (bounds[0][0] < root_pos_w[:, 0]) & (root_pos_w[:, 0] < bounds[0][1]) &
        (bounds[1][0] < root_pos_w[:, 1]) & (root_pos_w[:, 1] < bounds[1][1]) &
        (bounds[2][0] < root_pos_w[:, 2]) & (root_pos_w[:, 2] < bounds[2][1])
    )
    return ~torch.all(condition)

def finish_task(env: ManagerBasedRLEnv):
    final_dist = torch.linalg.vector_norm(
        env.current_state[:, 19:]-task_point,
        ord=2
    )
    return final_dist < thresholdCfg["task_finished"]
    
def normalize_observation(obs):
    return (obs - states_mid) / states_range

def inormalize_observation(norm_obs):
    return norm_obs * states_range + states_mid
    
def inormalize_action(norm_action):
    return norm_action * action_range + action_mid

def deal_obs(observation, num_envs):
    return torch.stack([normalize_observation(observation[i]) for i in range(num_envs)]).float()


class ActionClass(ActionTerm):

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

    def apply_actions(self) -> None:
        actions = self._processed_actions

        # print(f"-------a:{actions}")

        robot = self._asset
        joint_index = [robot.joint_names.index(j) for j in jointNames]
        obj = self._env.scene["objective"]
        ee_index = robot.body_names.index(eeName)
        ee_pos = robot.data.body_pos_w[:, ee_index]

        robot.write_root_velocity_to_sim(actions[:, 7:13])
        robot.write_joint_state_to_sim(actions[:, 13:16].float(), actions[:, 16:19].float(), joint_index)
        if self._env.is_catch:
            follow = [torch.stack([(ee_pos[i] + torch.tensor([0.05, 0, -0.01])), torch.tensor([0, 0, 0, 0])]) for i in range(self.num_envs)]
            obj.write_root_pose_to_sim(follow)
        

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._raw_actions = actions
        self._processed_actions = torch.stack([inormalize_action(actions[i]) for i in range(self.num_envs)]).double()

    @property
    def action_dim(self) -> int:
        return 19

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    

class MySb3VecEnvWrapper(Sb3VecEnvWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        observation_space = env.observation_space
        action_space = env.action_space
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)