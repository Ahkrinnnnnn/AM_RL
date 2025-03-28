import torch
from isaaclab.envs import ManagerBasedRLEnv

from AM_RL.Controller.cfgs.robotCfg import *

rewardsWeightCfg = {
    "ee_dist": -100,
    "time": -2,
    "is_captured": 1,
    "collision": -1000,
    "task_dist": -50,
    "pos_diff": -0.1,
    "alg_diff": -0.2,
    "vel_diff": -0.2,
    "joint_diff": -0.1,
    "joint_vel_diff": -0.2
}
thresholdCfg = {
    "track": 0.2,
    "task_finished": 0.1
}
task_point = torch.tensor([0, 0, 0.5], device="cuda")


##
# Customize functions
##


def get_ee_pos(env: ManagerBasedRLEnv, asset_name: str, ee_name: str):
    ee_index = env.scene[asset_name].body_names.index(ee_name)
    return env.scene[asset_name].data.body_pos_w[:, ee_index]


def ee_dist_reward(env: ManagerBasedRLEnv):
    """The closer the end-effector is to the objective within two timesteps, the higher the reward."""
    last_distance = torch.linalg.norm(env.last_ee - env.last_state[:, 19:22], dim=1, ord=2)
    current_distance = torch.linalg.norm(get_ee_pos(env, "uam", eeName) - env.current_state[:, 19:22], dim=1, ord=2)
    return (current_distance - last_distance) * rewardsWeightCfg["ee_dist"]

def time_reward(env: ManagerBasedRLEnv):
    """Penalty applied for each timestep to encourage faster task completion."""
    return rewardsWeightCfg["time"] * env.num_envs
 
def is_captured_reward(env: ManagerBasedRLEnv):
    """Reward for continuous carrying of the object."""
    return rewardsWeightCfg["is_captured"] * env.is_catch

#def collision_reward(env: ManagerBasedRLEnv, asset_name: str):
#    return env.scene[asset_name].is_in_collision() * rewardsWeightCfg["collision"]

def task_dist_reward(env: ManagerBasedRLEnv):
    """Reward for reducing the distance between the objective and the task point."""
    task_dist = torch.linalg.norm(env.current_state[:, 19:22]-task_point, dim=1, ord=2) - torch.linalg.norm(env.last_state[:, 19:22]-task_point, dim=1, ord=2)
    return task_dist * rewardsWeightCfg["task_dist"]

def plan_diff_reward(env: ManagerBasedRLEnv):
    """Penalty applied for mismatch between planned and actual position."""
    plan_diff = env.current_state[:, :19] - env.last_state[:, 22:]
    plan_diff = torch.where(torch.abs(plan_diff) < thresholdCfg["track"], torch.tensor(0.0, device=plan_diff.device), plan_diff)
    pos_diff = torch.linalg.norm(plan_diff[:, :3]) * rewardsWeightCfg["pos_diff"]
    alg_diff = torch.linalg.norm(plan_diff[:, 3:7]) * rewardsWeightCfg["alg_diff"]
    vel_diff = torch.linalg.norm(plan_diff[:, 7:13]) * rewardsWeightCfg["vel_diff"]
    joint_diff = torch.linalg.norm(plan_diff[:, 13:16]) * rewardsWeightCfg["joint_diff"]
    joint_vel_diff = torch.linalg.norm(plan_diff[:, 16:19]) * rewardsWeightCfg["joint_vel_diff"]
    return  pos_diff + alg_diff + vel_diff + joint_diff + joint_vel_diff
