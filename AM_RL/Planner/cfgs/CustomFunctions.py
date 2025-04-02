import os
import math

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaacsim.core.prims import Articulation
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.core.utils.rotations import euler_angles_to_quat
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import AM_RL
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
    return ~condition

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

def set_kinematics_solver(robot_prim_path, yaml_path, urdf_path, end_effector_name):
    robot = Articulation(robot_prim_path)
    kinematics_solver = LulaKinematicsSolver(
        robot_description_path=yaml_path,
        urdf_path=urdf_path
    )
    articulation_kinematics_solver = ArticulationKinematicsSolver(robot, kinematics_solver, end_effector_name)
    return kinematics_solver, articulation_kinematics_solver

def get_end_effector_world_pose(k_solver, ak_solver, joint_pos, robot_pos, robot_quat):
    k_solver.set_robot_base_pose(robot_pos, robot_quat)
    ee_position, ee_rotation_matrix = ak_solver.compute_end_effector_pose(joint_pos)
    return ee_position, ee_rotation_matrix

def calculate_yaw_angle(current_quat, target_quat):
    # Convert quaternions to Euler angles
    current_yaw, _, _ = euler_angles_to_quat(current_quat)
    target_yaw, _, _ = euler_angles_to_quat(target_quat)

    # Calculate yaw angle difference
    yaw_angle_diff = target_yaw - current_yaw

    # Normalize the yaw angle difference to be within -180 to 180 degrees
    yaw_angle_diff = math.degrees(yaw_angle_diff)
    yaw_angle_diff = (yaw_angle_diff + 180) % 360 - 180

    return yaw_angle_diff


class MySb3VecEnvWrapper(Sb3VecEnvWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        observation_space = env.observation_space
        action_space = env.action_space
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)