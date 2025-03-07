import random

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import (
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

from AM_RL.planner.cfgs import CustomFunctions
from AM_RL.planner.cfgs.robotCfg import *
import AM_RL.planner.cfgs.rewardCfg as RewardFunctions

##
# Scene definition
##


@configclass
class UamSceneCfg(InteractiveSceneCfg):
    """Configuration for a UAM scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0))
    )

    uam: ArticulationCfg = UAM_CFG

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)
    )

    objective = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/objective",
        spawn=sim_utils.UsdFileCfg(
            usd_path=rootPath+"/assets/usd/Basketball.usdz",
            scale=1,
            copy_from_source=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
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
        asset_name="uam"
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_pos = ObsTerm(func=mdp.observations.root_pos_w, params={"asset_cfg": SceneEntityCfg("uam")})
        base_quat = ObsTerm(func=mdp.observations.root_quat_w, params={"asset_cfg": SceneEntityCfg("uam")})
        base_lin_vel = ObsTerm(func=mdp.observations.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("uam")})
        base_ang_vel = ObsTerm(func=mdp.observations.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("uam")})
        joint_pos = ObsTerm(func=mdp.observations.joint_pos, params={"asset_cfg": SceneEntityCfg("uam", joint_names=jointNames)})
        joint_vel = ObsTerm(func=mdp.observations.joint_vel, params={"asset_cfg": SceneEntityCfg("uam", joint_names=jointNames)})
        obj_pos = ObsTerm(func=mdp.observations.root_pos_w, params={"asset_cfg": SceneEntityCfg("objective")})

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
        weight=1,
        params={"asset_name": "uam", "ee_name": eeName}
    )
    time = RewTerm(func=RewardFunctions.time_reward, weight=1)
    is_captured = RewTerm(
        func=RewardFunctions.is_captured_reward,
        weight=1,
        params={"asset_name": "uam", "ee_name": eeName}
    )
    collision = RewTerm(func=RewardFunctions.collision_reward, weight=1)
    smooth = RewTerm(func=RewardFunctions.smoothness_reward, weight=1)
    task_dist = RewTerm(func=RewardFunctions.task_dist_reward, weight=1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.terminations.time_out, time_out=True)
    # (2) AM out of bounds
    am_out_of_bounds = DoneTerm(
        func=CustomFunctions.robot_out_of_bounds,
        params={
            "asset_cfg": "uam",
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

    def __init__(self, cfg: UamEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.last_state = None
        self.current_state = None
        self.is_catch = False
    
    def reset(self):
        observation = super().reset()
        self.current_state = observation
        return CustomFunctions.normalize_observation(observation)

    def step(self, action):
        self.last_state = self.current_state

        action = CustomFunctions.inormalize_action(action)
        observation, reward, terminated, truncated, info = super().step(action)
        self.current_state = observation
        self.is_catch = self.is_catch or torch.linalg.norm(observation[:3]-observation[19:], ord=2) < 0.2

        return CustomFunctions.normalize_observation(observation), reward, terminated, truncated, info