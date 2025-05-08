import os
import math

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaacsim.core.utils.rotations import quat_to_euler_angles
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import AM_RL

##
# Customize variables
##


norm_path = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/Planner/model/pnorm_params.npz"
norm_params = np.load(norm_path)

states_mid = torch.tensor(norm_params["states_mid"], device="cuda")
states_range = torch.tensor(norm_params["states_range"], device="cuda")
action_mid = torch.tensor(norm_params["action_mid"], device="cuda")
action_range = torch.tensor(norm_params["action_range"], device="cuda")

task_point = torch.tensor([0, 0, 0.5], device="cuda")


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
        env.current_state[:, 10:]-task_point,
        ord=2
    )
    return final_dist < 0.1
    
def normalize_observation(obs):
    return (obs - states_mid) / states_range

def inormalize_observation(norm_obs):
    return norm_obs * states_range + states_mid
    
def inormalize_action(norm_action):
    return norm_action * action_range + action_mid

def deal_obs(observation, num_envs):
    return torch.stack([normalize_observation(observation[i]) for i in range(num_envs)]).float()

# computer ee_pos
def euler_matrix(roll, pitch, yaw, device='cpu'):
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ], device=device)

    Ry = torch.tensor([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ], device=device)

    Rz = torch.tensor([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ], device=device)
    
    return Rz @ Ry @ Rx

def rotation_matrix(theta, axis, device='cpu'):
    axis = axis / torch.linalg.norm(axis)
    a = axis[0]
    b = axis[1]
    c = axis[2]
    
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack([
        torch.stack([a*a*(1-cos)+cos, a*b*(1-cos)-c*sin, a*c*(1-cos)+b*sin], dim=-1),
        torch.stack([a*b*(1-cos)+c*sin, b*b*(1-cos)+cos, b*c*(1-cos)-a*sin], dim=-1),
        torch.stack([a*c*(1-cos)-b*sin, b*c*(1-cos)+a*sin, c*c*(1-cos)+cos], dim=-1)
    ], dim=-2)

def get_end_effector_world_pose(thetas, device='cpu'):
    batch_size = thetas.size(0)
    T = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    
    T_base = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    T_base[:, 2, 3] = -0.10
    T = T @ T_base

    # joint1
    R_origin1 = euler_matrix(1.5708, 1.5708, 0, device)
    R_origin1 = R_origin1.unsqueeze(0).repeat(batch_size, 1, 1)
    
    R_joint1 = rotation_matrix(thetas[:, 0], torch.tensor([0, 0, 1], device=device, dtype=torch.float64))
    T_joint1 = torch.cat([
        torch.cat([R_origin1 @ R_joint1, torch.zeros(batch_size, 3, 1, device=device)], dim=-1),
        torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(batch_size,1,1)
    ], dim=-2)
    T = T @ T_joint1

    # joint2
    T_trans2 = torch.eye(4, device=device).repeat(batch_size,1,1)
    T_trans2[:, 0, 3] = 0.132
    R_joint2 = rotation_matrix(thetas[:, 1], torch.tensor([0, 0, 1], device=device, dtype=torch.float64))
    T_joint2 = T_trans2 @ torch.cat([
        torch.cat([R_joint2, torch.zeros(batch_size,3,1, device=device)], dim=-1),
        torch.tensor([[[0,0,0,1]]], device=device).repeat(batch_size,1,1)
    ], dim=-2)
    T = T @ T_joint2

    # joint3
    T_trans3 = torch.eye(4, device=device).repeat(batch_size,1,1)
    T_trans3[:, 0, 3] = 0.075
    R_joint3 = rotation_matrix(thetas[:, 2], torch.tensor([0, 0, 1], device=device, dtype=torch.float64))
    T_joint3 = T_trans3 @ torch.cat([
        torch.cat([R_joint3, torch.zeros(batch_size,3,1, device=device)], dim=-1),
        torch.tensor([[[0,0,0,1]]], device=device).repeat(batch_size,1,1)
    ], dim=-2)
    T = T @ T_joint3

    # flying_arm_3__j_link_3_gripper
    T_gripper = torch.eye(4, device=device).repeat(batch_size,1,1)
    T_gripper[:, 0, 3] = 0.05
    T_gripper[:, 2, 3] = -0.01
    T = T @ T_gripper

    return T[:, :3, 3]

# for control
def calculate_yaw_angle(current_quat, target_quat):
    dev = current_quat.device
    current_quat = current_quat.cpu().numpy()
    target_quat = target_quat.cpu().numpy()

    # Convert quaternions to Euler angles
    current_yaw = torch.tensor(np.array([quat_to_euler_angles(current_quat[i]) for i in range(len(current_quat))]), device=dev)[:, 0]
    target_yaw = torch.tensor(np.array([quat_to_euler_angles(target_quat[i]) for i in range(len(target_quat))]), device=dev)[:, 0]

    # Calculate yaw angle difference
    yaw_angle_diff = target_yaw - current_yaw

    # Normalize the yaw angle difference to be within -180 to 180 degrees
    yaw_angle_diff = torch.tensor([math.degrees(yaw_angle_diff[i]) for i in range(len(yaw_angle_diff))], device=dev)
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