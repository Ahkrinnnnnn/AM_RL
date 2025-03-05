import torch
from isaaclab.envs import ManagerBasedRLEnv

rewardsWeightCfg = {
    "ee_dist": -1,
    "time": -1,
    "is_captured": 50,
    "collision": -1000,
    "heading": 2,
    "angle": 2,
    "task_dist": -2
}
thresholdCfg = {
    "heading_thresh": 0.1,
    "angle_thresh": 0.1,
    "task_finished": 0.1
}
task_point = torch.tensor[0, 0, 0.5]

class RewardFunctions:

    @staticmethod
    def ee_dist_reward(env: ManagerBasedRLEnv, asset_name: str, ee_name: str):
        ee_index = env.scene[asset_name].joint_names.index(ee_name)
        ee_pos = env.scene[asset_name].data.joint_pos[ee_index]
        last_distance = torch.linalg.norm(ee_pos - env.last_state[19:], ord=2)
        current_distance = torch.linalg.norm(ee_pos - env.current_state[19:], ord=2)
        return (current_distance - last_distance) * rewardsWeightCfg["ee_dist"]
    
    @staticmethod
    def time_reward(env: ManagerBasedRLEnv):
        return rewardsWeightCfg["time"]
    
    @staticmethod
    def is_captured_reward(env: ManagerBasedRLEnv, asset_name: str, ee_name: str, threshold: float = 0.1):
        ee_index = env.scene[asset_name].joint_names.index(ee_name)
        ee_pos = env.scene[asset_name].data.joint_pos[ee_index]
        current_distance = torch.linalg.norm(ee_pos - env.current_state[19:], ord=2)
        if current_distance < threshold:
            return rewardsWeightCfg["is_captured"]
        else:
            return 0.0
    
    @staticmethod
    def collision_reward(env: ManagerBasedRLEnv, asset_name: str):
        if env.scene[asset_name].is_in_collision():
            return rewardsWeightCfg["collision"]
        return 0.0

    @staticmethod
    def smoothness_reward(env: ManagerBasedRLEnv):
        dx = env.current_state[:3] - env.last_state[:3]
        vt = env.current_state[7:10] * env.sim.dt
        dang = env.current_state[3:7] - env.last_state[3:7]
        heading_reward = -1 * rewardsWeightCfg["heading"]
        if torch.all(torch.linalg.norm(dx - vt, ord=1) < thresholdCfg["heading_thresh"]):
            heading_reward = rewardsWeightCfg["heading"]
        ang_reward = -1 * rewardsWeightCfg["angle"]
        if torch.all(dang < thresholdCfg["angle_thresh"]):
            ang_reward = rewardsWeightCfg["heading"]
        return heading_reward + ang_reward
        
    @staticmethod
    def task_dist_reward(env: ManagerBasedRLEnv):
        task_dist = torch.linalg.norm(env.current_state[19:]-task_point, ord=2) - torch.linalg.norm(env.last_state[19:]-task_point, ord=2)
        return task_dist * rewardsWeightCfg["task_dist"]
