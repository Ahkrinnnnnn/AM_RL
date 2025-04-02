import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg

from AM_RL.Planner.cfgs.robotCfg import *
from AM_RL.Planner.cfgs.control import pd_control
import AM_RL.Planner.cfgs.CustomFunctions as CustomFunctions

class ActionClass(ActionTerm):

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

    def apply_actions(self) -> None:
        actions = self._processed_actions

        # print(f"-------a:{actions}")

        robot = self._asset
        joint_index = [robot.joint_names.index(j) for j in jointNames]
        base_link_index = robot.body_names.index(baseLinkName)
        obj = self._env.scene["objective"]

        force, torque = pd_control(
            self._env.current_state[:, :3],
            robot.data.root_lin_vel_w,
            robot.data.root_ang_vel_w,
            actions[:, :3],
            CustomFunctions.calculate_yaw_angle(
                self._env.current_state[:, 3:7],
                actions[:, 3:7]
            )
        )
        robot.set_external_force_and_torque(force, torque, base_link_index)
        robot.set_joint_position_target(actions[:, 7:].float(), joint_index)
        robot.write_data_to_sim()

        for i in range(self.num_envs):
            if self._env.is_catch[i]:
                obj.write_root_pose_to_sim(torch.cat([
                    self._env.current_ee_pos[i],
                    torch.tensor([0, 0, 0, 1], device=self._env.device)
                ]), torch.tensor([i], device=self._env.device))

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self._raw_actions = actions
        self._processed_actions = torch.stack([CustomFunctions.inormalize_action(actions[i]) for i in range(self.num_envs)]).double()

    @property
    def action_dim(self) -> int:
        return 10

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions