import os
import random

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

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


class CustomFunctions:

    @staticmethod
    def robot_out_of_bounds(env: ManagerBasedRLEnv, asset_name: str, bounds: list):
        x, y, z = env.scene[asset_name].data.root_pos_w
        
        if (bounds[0][0] < x < bounds[0][1] and 
            bounds[1][0] < y < bounds[1][1] and
            bounds[2][0] < z < bounds[2][1]):
            return False
        else:
            return True

    @staticmethod
    def finish_task(asset_cfg: SceneEntityCfg):
        final_dist = torch.linalg.vector_norm(
            mdp.observations.root_pos_w(asset_cfg)-task_point,
            ord=2
        )
        return final_dist < thresholdCfg["task_finished"]
    
    @staticmethod
    def normalize_observation(obs):
        return (obs - states_mid) / states_range
    
    @staticmethod
    def inormalize_observation(norm_obs):
        return norm_obs * states_range + states_mid
    
    @staticmethod
    def inormalize_action(norm_action):
        return norm_action * action_range + action_mid
    

    class ActionClass(ActionTerm):
        def __init__(self, asset_name: str, params: dict):
            super().__init__()
            self.asset_name = asset_name
            self.params = params

        def execute(self, action):
            action = CustomFunctions.inormalize_action(action)
            robot = self.env.scene[self.asset_name]
            robot.write_root_link_state_to_sim(action[:13])
            robot.write_joint_state_to_sim(action[13:16], action[16:19], jointNames)


##
# Scene definition
##


@configclass
class UamSceneCfg(InteractiveSceneCfg):
    """Configuration for a UAM scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0))
    )

    robot: ArticulationCfg = UAM_CFG

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)
    )

    objective = AssetBaseCfg(
        prim_path="/World/Sphere",
        spawn=sim_utils.SphereCfg(radius=0.2),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=((random.random()-0.5)*20, (random.random()-0.5)*20, 0.1)
        )
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    planning_state = ActionTermCfg(
        class_type=CustomFunctions.ActionClass,
        asset_name={"asset_name": "robot"}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pos = ObsTerm(func=mdp.observations.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_quat = ObsTerm(func=mdp.observations.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func=mdp.observations.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.observations.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp.observations.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=jointNames)})
        joint_vel = ObsTerm(func=mdp.observations.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=jointNames)})
        obj_pos = ObsTerm(func=mdp.observations.root_pos_w, params={"asset_cfg": SceneEntityCfg("objective")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.normalize_fn = CustomFunctions.normalize_observation

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_env = EventTerm(
        func=mdp.events.reset_scene_to_default,
        mode="reset"
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ee_dist = RewTerm(
        func=RewardFunctions.ee_dist_reward,
        params={"asset_name": "robot", "ee_name": eeName}
    )
    time = RewTerm(func=RewardFunctions.time_reward)
    is_captured = RewTerm(
        func=RewardFunctions.is_captured_reward,
        params={"asset_name": "robot", "ee_name": eeName}
    )
    collision = RewTerm(func=RewardFunctions.collision_reward)
    smooth = RewTerm(func=RewardFunctions.smoothness_reward)
    task_dist = RewTerm(func=RewardFunctions.task_dist_reward)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.terminations.time_out, time_out=True)
    # (2) AM out of bounds
    am_out_of_bounds = DoneTerm(
        func=CustomFunctions.robot_out_of_bounds,
        params={
            "asset_cfg": "robot",
            "bounds": [[-10, 10], [-10, 10], [0, 10]]
        }
    )
    # (3) Task finished
    task_finished = DoneTerm(
        func=CustomFunctions.finish_task,
        params={"asset_cfg": SceneEntityCfg("objective")}
    )


##
# Environment configuration
##


@configclass
class UamEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the UAM environment."""

    # Scene settings
    scene: UamSceneCfg = UamSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


##
# Customize environment
##


class CustomEnv(ManagerBasedRLEnv):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.last_state = None
        self.current_state = None
    
    def reset(self):
        observation = super().reset()
        self.current_state = CustomFunctions.inormalize_observation(observation)
        return observation

    def step(self, action):
        self.last_state = self.current_state

        observation, reward, terminated, truncated, info = super().step(action)
        self.current_state = CustomFunctions.inormalize_observation(observation)

        return observation, reward, terminated, truncated, info