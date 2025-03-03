import random

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from robotCfg import *
from rewardCfg import RewardFunctions

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

    joint_effort = mdp.actions.JointEffortActionCfg(asset_name="robot", joint_names=rotorNames+jointNames)


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
    track_dis = RewTerm(
        func=RewardFunctions.track_dist_reward,
        params={"asset_name": "robot"}
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
    # am_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )


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
