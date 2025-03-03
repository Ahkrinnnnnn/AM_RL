import numpy as np
from isaaclab.envs import ManagerBasedRLEnv

rewardsWeightCfg = {
    "ee_dist": -1,
    "track_dist": -0.5,
    "traj_thresh": 0.2,
    "time": -1,
    "is_captured": 100,
    "collision": -1000
}

class RewardFunctions:

    def ee_dist_reward(env: ManagerBasedRLEnv, asset_name: str, target_name: str):
        ee_pos = env.scene[asset_name].data.root_pos_w
        target_pos = env.scene[target_name].data.root_pos_w
        distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
        return distance * rewardsWeightCfg["ee_dist"]
    
    def track_dist_reward(env: ManagerBasedRLEnv, asset_name: str):
        robot_pos = env.scene[asset_name].data.root_pos_w
        distance = np.linalg.norm(np.array(robot_pos) - np.array(env.planed_point))
        return distance * rewardsWeightCfg["track_dist"]
    
    def time_reward(env: ManagerBasedRLEnv):
        return rewardsWeightCfg["time"]
    
    def is_captured_reward(env: ManagerBasedRLEnv, asset_name: str, target_name: str, threshold: float = 0.1):
        ee_pos = env.scene[asset_name].data.root_pos_w
        target_pos = env.scene[target_name].data.root_pos_w
        distance = np.linalg.norm(np.array(ee_pos) - np.array(target_pos))
        if distance < threshold:
            return rewardsWeightCfg["is_captured"]
        else:
            return 0.0
    
    def collision_reward(env: ManagerBasedRLEnv, asset_name: str):
        if env.scene[asset_name].is_in_collision():
            return rewardsWeightCfg["collision"]
        return 0.0
