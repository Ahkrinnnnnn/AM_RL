import torch
import numpy as np
from gymnasium import spaces
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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

from stable_baselines3 import TD3

from AM_RL.Controller.cfgs.robotCfg import *
import AM_RL.Controller.cfgs.CustomFunctions as CustomFunctions
import AM_RL.Controller.cfgs.rewardCfg as RewardFunctions
from AM_RL.Planner.model.planner import PlannerNetwork

##
# Scene definition
##


@configclass
class UamSceneCfg(InteractiveSceneCfg):
    """Configuration for a UAM scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(20.0, 20.0),
            color=(1.0, 1.0, 1.0)
        )
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9), 
            intensity=500.0
        )
    )

    uam: ArticulationCfg = UAM_CFG

    objective = obj_CFG


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    control = ActionTermCfg(
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
        weight=1
    )
    time = RewTerm(
        func=RewardFunctions.time_reward,
        weight=1
    )
    is_captured = RewTerm(
        func=RewardFunctions.is_captured_reward,
        weight=1
    )
    task_dist = RewTerm(
        func=RewardFunctions.task_dist_reward,
        weight=1
    )
    plan_diff = RewTerm(
        func=RewardFunctions.plan_diff_reward,
        weight=1
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
            "asset_name": "uam",
            "bounds": [[-10, 10], [-10, 10], [0, 10]]
        }
    )
    # (3) Task finished
    task_finished = DoneTerm(func=CustomFunctions.finish_task)


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

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(41,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
    
        self.last_state = None
        self.last_ee = None
        self.current_state = None
        self.is_catch = torch.tensor([False]*self.num_envs, device=self.device)

        self.plan_net = PlannerNetwork(
            spaces.Box(low=-1, high=1, shape=(22,), dtype=np.float32),
            spaces.Box(low=-1, high=1, shape=(19,), dtype=np.float32)
        )
        planner_model_path = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/Planner/model/2025-03-26_17-24-44/model_496000_steps.zip"
        planner = TD3.load(planner_model_path)
        self.plan_net.load_state_dict(planner.policy.actor.state_dict())
        self.plan_net.to(self.device)
        self.plan_net.eval()

    
    def reset(self, seed, options):
        observation = super().reset(seed=seed, env_ids=None, options=options)
        self.current_state = CustomFunctions.comb_obs(self, observation[0]['policy'])
        observation[0]['policy'] = CustomFunctions.deal_obs(self.current_state, self.num_envs)

        return observation

    def step(self, action):
        self.last_state = self.current_state
        self.last_ee = RewardFunctions.get_ee_pos(self, "uam", eeName)

        observation, reward, terminated, truncated, info = super().step(action)
        self.current_state = CustomFunctions.comb_obs(self, observation['policy'])
        self.is_catch = self.is_catch | (torch.linalg.norm(RewardFunctions.get_ee_pos(self, "uam", eeName)-self.current_state[:, 19:22], dim=1, ord=2) < 0.1)

        observation['policy'] = CustomFunctions.deal_obs(self.current_state, self.num_envs)
        return observation, reward, terminated, truncated, info