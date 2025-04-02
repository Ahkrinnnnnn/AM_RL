import torch
from isaaclab.envs import ManagerBasedRLEnv

from AM_RL.Planner.cfgs.robotCfg import k_solver, ak_solver
from AM_RL.Planner.cfgs.CustomFunctions import task_point, get_end_effector_world_pose

rewardsWeightCfg = {
    "ee_dist": -20,
    "time": -2,
    "is_captured": 1,
    "collision": -1000,
    "task_dist": -50,
    "plan_diff_pos": -0.5,
    "plan_diff_alg": -0.5
}


##
# Customize functions
##


def ee_dist_reward(env: ManagerBasedRLEnv):
    """The closer the end-effector is to the objective within two timesteps, the higher the reward."""
    last_distance = torch.linalg.norm(env.last_ee - env.last_state[:, 10:], dim=1, ord=2)
    ee_pos = get_end_effector_world_pose(
        k_solver, ak_solver, 
        env.current_state[:, 7:10], env.current_state[:, :3], env.current_state[:, 3:7]
    )
    current_distance = torch.linalg.norm(ee_pos-env.current_state[:, 10:], dim=1, ord=2)
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
    task_dist = torch.linalg.norm(env.current_state[:, 10:]-task_point, dim=1, ord=2) - torch.linalg.norm(env.last_state[:, 10:]-task_point, dim=1, ord=2)
    return task_dist * rewardsWeightCfg["task_dist"]

def plan_diff_reward(env: ManagerBasedRLEnv):
    """Penalty applied for mismatch between planned and actual position."""
    if env.planned == None:
        return 0
    plan_diff_pos = torch.linalg.norm(env.planned[:, :3] - env.current_state[:, :3], dim=1, ord=2)
    plan_diff_alg = torch.linalg.norm(env.planned[:, 3:7] - env.current_state[:, 3:7], dim=1, ord=2)
    return plan_diff_pos * rewardsWeightCfg["plan_diff_pos"] + plan_diff_alg * rewardsWeightCfg["plan_diff_alg"]
