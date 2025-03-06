import os

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, mdp
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg

import AM_RL
from AM_RL.planner.cfgs.robotCfg import *
from AM_RL.planner.cfgs.rewardCfg import *

norm_path = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/planner/model/pnorm_params.npz"
norm_params = np.load(norm_path)

states_mid = norm_params["states_mid"]
states_range = norm_params["states_range"]
action_mid = norm_params["action_mid"]
action_range = norm_params["action_range"]


##
# Customize functions
##


def robot_out_of_bounds(env: ManagerBasedRLEnv, asset_name: str, bounds: list):
    x, y, z = env.scene[asset_name].data.root_pos_w
        
    if (bounds[0][0] < x < bounds[0][1] and 
        bounds[1][0] < y < bounds[1][1] and
        bounds[2][0] < z < bounds[2][1]):
        return False
    else:
        return True

def finish_task(asset_cfg: SceneEntityCfg):
    final_dist = torch.linalg.vector_norm(
        mdp.observations.root_pos_w(asset_cfg)-task_point,
        ord=2
    )
    return final_dist < thresholdCfg["task_finished"]
    
def normalize_observation(obs):
    return (obs - states_mid) / states_range

def inormalize_observation(norm_obs):
    return norm_obs * states_range + states_mid
    
def inormalize_action(norm_action):
    return norm_action * action_range + action_mid
    

class ActionClass(ActionTerm):

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    @property
    def action_dim(self) -> int:
        return 19

    def apply_actions(self, actions: torch.Tensor) -> None:
        robot = self.env.scene[self.asset_name]
        robot.write_root_link_state_to_sim(actions[:13])
        robot.write_joint_state_to_sim(actions[13:16], actions[16:19], jointNames)
        obj = self.env.scene["objective"]
        ee_index = robot.joint_names.index(eeName)
        ee_pos = robot.data.joint_pos[ee_index]
        if self.env.is_catch:
            obj.write_root_link_state_to_sim(ee_pos + torch.tensor([0, 0, 0, 0]))

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def processed_actions(self) -> torch.Tensor:
        return self.process_actions(self.raw_actions())

    def raw_actions(self) -> torch.Tensor:
        return self.env.get_actions()
