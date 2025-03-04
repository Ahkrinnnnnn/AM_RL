import random

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from robotCfg import *
from rewardCfg import RewardFunctions

##
# Customize functions
##


class CustomFunctions:

    def robot_out_of_bounds(env: ManagerBasedRLEnv, asset_name: str, bounds: list):
        x, y, z = env.scene[asset_name].data.root_pos_w
        
        if (bounds[0][0] < x < bounds[0][1] and 
            bounds[1][0] < y < bounds[1][1] and
            bounds[2][0] < z < bounds[2][1]):
            return True
        else:
            return False

    def plan_next_point(env: ManagerBasedRLEnv):
        return env.plan_next_point()

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
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            pos=((random.random()-0.5)*20, (random.random()-0.5)*20, 0.1),
            color=(1.0, 0.0, 0.0),
            collision=True
        ),
        name="objective"
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    planning_state = ActionTerm(
        func=CustomFunctions.plan_next_point
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        obj_pos = ObsTerm(func=mdp.observations.root_pos_w, params={"asset_cfg": SceneEntityCfg("objective")})
        base_pos = ObsTerm(func=mdp.observations.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_quat = ObsTerm(func=mdp.observations.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func=mdp.observations.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.observations.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp.observations.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=jointNames)})
        joint_vel = ObsTerm(func=mdp.observations.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=jointNames)})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

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
        params={"asset_name": eeName,
                "target_name": "objective"}
    )
    time = RewTerm(func=RewardFunctions.time_reward)
    capture = RewTerm(
        func=RewardFunctions.is_captured_reward,
        params={"asset_name": "robot", "target_name": "objective", "threshold": 0.1}
    )
    collision = RewTerm(
        func=RewardFunctions.collision_reward,
        params={"asset_name": "robot"}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.terminations.time_out, time_out=True)
    # (2) AM out of bounds
    am_out_of_bounds = DoneTerm(
        func=CustomFunctions.robot_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bounds": [[-10, 10], [-10, 10], [0, 10]]
        }
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

    def __init__(self):
        super().__init__()

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

        self.next_state = None
    
    def reset(self):
        state = super().reset()
        self.last_state = state
        return state

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.next_state = self.plan_next_point(observation)

        return observation, reward, terminated, truncated, info
    
    def plan_next_point(self, observation):

        return 