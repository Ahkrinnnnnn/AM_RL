import os

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, mdp
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import AM_RL
from AM_RL.Controller.cfgs.robotCfg import *
from AM_RL.Controller.cfgs.rewardCfg import *

norm_path = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/Controller/model/cnorm_params.npz"
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
    return ~condition

def finish_task(env: ManagerBasedRLEnv):
    final_dist = torch.linalg.vector_norm(
        env.current_state[:, 19:22]-task_point,
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

def plan(env: ManagerBasedEnv, obs):
    with torch.no_grad():
        next_planning_state = torch.zeros([env.num_envs, 10], device=env.device)
        for i in range(env.num_envs):
            ob = torch.cat([obs[i][:7], obs[i][13:16], obs[i][19:22]])
            next_planning_state[i] = env.plan_net(ob)
    return next_planning_state

def comb_obs(env: ManagerBasedEnv, obs):
    return torch.cat([obs, plan(env, obs)], dim=1)


class ActionClass(ActionTerm):

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

    def apply_actions(self) -> None:
        actions = self._processed_actions

        # print(f"-------a:{actions}")

        robot = self._asset
        rotor_index = [robot.joint_names.index(j) for j in rotorNames]
        rotor_link_index = [robot.body_names.index(l) for l in rotorLinkNames]
        joint_index = [robot.joint_names.index(j) for j in jointNames]
        
        force = torch.cat([torch.zeros(actions.shape[0], 6, 2, device=self._env.device), (actions[:, :6] * 2).unsqueeze(-1)], dim=2)
        torque = torch.zeros_like(force)
        robot.set_external_force_and_torque(force, torque, rotor_link_index)
        robot.set_joint_effort_target(actions[:, 6:9], joint_index)
        robot.write_data_to_sim()

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._raw_actions = actions
        self._processed_actions = torch.stack([inormalize_action(actions[i]) for i in range(self.num_envs)]).float()

    @property
    def action_dim(self) -> int:
        return 9

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