import os

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, mdp
from isaaclab.managers import (
    ActionTerm, ActionTermCfg, SceneEntityCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)
from isaaclab.utils import configclass

import AM_RL
from AM_RL.Planner.cfgs.robotCfg import *
from AM_RL.Planner.cfgs.rewardCfg import *

norm_path = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/Planner/model/pnorm_params.npz"
norm_params = np.load(norm_path)

states_mid = torch.tensor(norm_params["states_mid"], device="cuda")
states_range = torch.tensor(norm_params["states_range"], device="cuda")
action_mid = torch.tensor(norm_params["action_mid"], device="cuda")
action_range = torch.tensor(norm_params["action_range"], device="cuda")

state_clip = ((states_mid-states_range/2).cpu().numpy(), (states_mid+states_range/2).cpu().numpy())

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
    return torch.all(condition)

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
    return torch.stack([inormalize_observation(observation[i]) for i in range(num_envs)])


class ActionClass(ActionTerm):

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

    def apply_actions(self) -> None:
        actions = self._processed_actions
        robot = self._asset
        joint_index = [robot.joint_names.index(j) for j in jointNames]
        obj = self._env.scene["objective"]
        ee_index = robot.body_names.index(eeName)
        ee_pos = robot.data.body_pos_w[:, ee_index]

        robot.write_root_state_to_sim(actions[:, :13])
        robot.write_joint_state_to_sim(actions[:, 13:16].float(), actions[:, 16:19].float(), joint_index)
        if self._env.is_catch:
            follow = [torch.stack([(ee_pos[i] + torch.tensor([0.05, 0, -0.01])), torch.tensor([0, 0, 0, 0])]) for i in range(self.num_envs)]
            obj.write_root_pose_to_sim(follow)

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._raw_actions = actions
        # print(f"------------{actions}")
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
    

@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_pos = ObsTerm(
        func=mdp.observations.root_pos_w,
        params={"asset_cfg": SceneEntityCfg("uam")},
        clip=(state_clip[0][0], state_clip[1][0]),
        scale=tuple(1 / states_range[:3]),
    )
    base_quat = ObsTerm(
        func=mdp.observations.root_quat_w, 
        params={"asset_cfg": SceneEntityCfg("uam")},
        clip=(state_clip[0][3], state_clip[1][3]),
        scale=tuple(1 / states_range[3:7])
    )
    base_lin_vel = ObsTerm(
        func=mdp.observations.root_lin_vel_w,
        params={"asset_cfg": SceneEntityCfg("uam")},
        clip=(state_clip[0][7], state_clip[1][7]),
        scale=tuple(1 / states_range[7:10])
    )
    base_ang_vel = ObsTerm(
        func=mdp.observations.root_ang_vel_w, 
        params={"asset_cfg": SceneEntityCfg("uam")},
        clip=(state_clip[0][10], state_clip[1][10]),
        scale=tuple(1 / states_range[10:13])
    )
    joint_pos = ObsTerm(
        func=mdp.observations.joint_pos, 
        params={"asset_cfg": SceneEntityCfg("uam", joint_names=jointNames)},
        clip=(state_clip[0][13], state_clip[1][13]),
        scale=tuple(1 / states_range[13:16])
    )
    joint_vel = ObsTerm(
        func=mdp.observations.joint_vel, 
        params={"asset_cfg": SceneEntityCfg("uam", joint_names=jointNames)},
        clip=(state_clip[0][16], state_clip[1][16]),
        scale=tuple(1 / states_range[16:19])
    )
    obj_pos = ObsTerm(
        func=mdp.observations.root_pos_w, 
        params={"asset_cfg": SceneEntityCfg("objective")},
        clip=(state_clip[0][19], state_clip[1][19]),
        scale=tuple(1 / states_range[19:])
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True