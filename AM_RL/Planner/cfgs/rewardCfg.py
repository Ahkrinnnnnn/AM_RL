import torch
from isaaclab.envs import ManagerBasedRLEnv

from AM_RL.Planner.cfgs.robotCfg import *

rewardsWeightCfg = {
    "ee_dist": -20,
    "time": -2,
    "is_captured": 1,
    "collision": -1000,
    "heading": 10, #
    "angle": 10, #
    "task_dist": -50,
    "plan_diff_pos": -0.5,
    "plan_diff_alg": -0.5
}
thresholdCfg = {
    "heading_thresh": 1,
    "angle_thresh": 1,
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
    last_distance = torch.linalg.norm(env.last_ee - env.last_state[:, 19:], dim=1, ord=2)
    current_distance = torch.linalg.norm(get_ee_pos(env, "uam", eeName) - env.current_state[:, 19:], dim=1, ord=2)
    return (current_distance - last_distance) * rewardsWeightCfg["ee_dist"]

def time_reward(env: ManagerBasedRLEnv):
    """Penalty applied for each timestep to encourage faster task completion."""
    return rewardsWeightCfg["time"] * env.num_envs
 
def is_captured_reward(env: ManagerBasedRLEnv):
    """Reward for continuous carrying of the object."""
    return rewardsWeightCfg["is_captured"] * env.is_catch

#def collision_reward(env: ManagerBasedRLEnv, asset_name: str):
#    return env.scene[asset_name].is_in_collision() * rewardsWeightCfg["collision"]

def smoothness_reward(env: ManagerBasedRLEnv):
    """Limit the velocity derivative to avoid excessive values."""
    dv = (env.current_state[:, 7:10] - env.last_state[:, 7:10]) / env.step_dt
    dang = (env.current_state[:, 10:13] - env.last_state[:, 10:13]) / env.step_dt
    lin_reward = -1 * rewardsWeightCfg["heading"]
    if torch.all(torch.linalg.norm(dv, dim=1, ord=2) < thresholdCfg["heading_thresh"]):
        lin_reward = rewardsWeightCfg["heading"]
    ang_reward = -1 * rewardsWeightCfg["angle"]
    if torch.all(torch.linalg.norm(dang, dim=1, ord=2) < thresholdCfg["angle_thresh"]):
        ang_reward = rewardsWeightCfg["heading"]
    return (lin_reward + ang_reward) * env.num_envs

def task_dist_reward(env: ManagerBasedRLEnv):
    """Reward for reducing the distance between the objective and the task point."""
    task_dist = torch.linalg.norm(env.current_state[:, 19:]-task_point, dim=1, ord=2) - torch.linalg.norm(env.last_state[:, 19:]-task_point, dim=1, ord=2)
    return task_dist * rewardsWeightCfg["task_dist"]

def plan_diff_reward(env: ManagerBasedRLEnv):
    """Penalty applied for mismatch between planned and actual position."""
    if env.planned == None:
        return 0
    plan_diff_pos = torch.linalg.norm(env.planned[:, :3] - env.current_state[:, :3], dim=1, ord=2)
    plan_diff_alg = torch.linalg.norm(env.planned[:, 3:7] - env.current_state[:, 3:7], dim=1, ord=2)
    return plan_diff_pos * rewardsWeightCfg["plan_diff_pos"] + plan_diff_alg * rewardsWeightCfg["plan_diff_alg"]
