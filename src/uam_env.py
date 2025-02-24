import genesis as gs
import random
import torch

class UamEnv:
    def __init__(self, urdf, work_area, show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        # self.num_envs = num_envs
        self.work_area = work_area

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options = gs.options.ViewerOptions(
                res           = (960, 960),
                camera_pos    = (3.5, 0.0, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 40,
                max_FPS       = 90,
            ),
        )

        self.uam_cam = self.scene.add_camera(
            res=(1280, 960), 
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=True
        )

        self.scene.add_entity(gs.morphs.Plane(collision=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf, 
                pos=(0, 0, 0.2),
                quat=(0, 0, 0, 1),
                collision=True,
                prioritize_urdf_material=True
            )
        )


        self.obj = self.scene.add_entity(
            gs.morphs.Sphere(
                pos=(
                    random.uniform(self.work_area[0][0], self.work_area[0][1]), 
                    random.uniform(self.work_area[1][0], self.work_area[1][1]), 
                    0.05
                ),
                radius=0.05,
                collision=True
            )
        )
        self.obstacles = [
            self.scene.add_entity(
                gs.morphs.Box(
                    pos=(
                        random.uniform(self.work_area[0][0], self.work_area[0][1]), 
                        random.uniform(self.work_area[1][0], self.work_area[1][1]), 
                        0.5
                    ),
                    size=(0.5, 0.5, 1)
                )
            ),
            self.scene.add_entity(
                gs.morphs.Box(
                    pos=(
                        random.uniform(self.work_area[0][0], self.work_area[0][1]), 
                        random.uniform(self.work_area[1][0], self.work_area[1][1]), 
                        0.15
                    ),
                    size=(0.3, 0.3, 0.3)
                )
            )
        ]

        # self.end_effector = self.robot.get_link("flying_arm_3__gripper")
        self.end_effector = self.robot.get_link("flying_arm_3__link_3")
        self.get_idx()
        # self.scene.build(n_envs=self.num_envs, env_spacing=(3.0, 3.0))
        self.scene.build()

    def get_idx(self):
        motorNames = ["hexacopter370__rotor_" + str(i) for i in range(6)]
        self.motors_idx = [self.robot.get_link(name).idx_local for name in motorNames]
        rotorNames = ["hexacopter370__rotor_" + str(i) + "_joint" for i in range(6)]
        self.rotors_radius = 0.0889
        self.rotors_idx = [self.robot.get_joint(name).dof_idx_local for name in rotorNames]
        jointNames = ["flying_arm_3__j_base_link_link_1",
                    "flying_arm_3__j_link_1_link_2",
                    "flying_arm_3__j_link_2_link_3"]
        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in jointNames]

    def step(self, action):
        # states = self.get_full_states()
        # print(states.round(decimals=2))
        self.control_output(self.robot, action[:6], action[6:])

        self.scene.step()

    def get_full_states(self):
        base = self.robot.base_link
        states = torch.cat([
            base.get_pos(), 
            base.get_quat(),
            base.get_vel(), 
            base.get_ang(),
            self.robot.get_dofs_position(self.dofs_idx),
            self.robot.get_dofs_velocity(self.dofs_idx)
        ], dim=0)
        return states

    def get_img(self):
        self.uam_cam.set_pose(pos=self.robot.get_pos()+[0.21, 0, 0], quat=self.robot.get_quat())
        rgb, depth, segmentation, normal = self.uam_cam.render(depth=True, segmentation=True, normal=True)
        return torch.tensor(rgb)

    def control_output(self, motors_force, joints_torque):
        self.robot.control_dofs_force(motors_force, self.motors_idx)
        self.robot.control_dofs_force(joints_torque, self.dofs_idx)
        # propellerTorque = motors_force * self.rotors_radius / 2.0  * torch.tensor(
        #         [(-1)**i for i in range(len(self.rotors_idx))]
        #     ) * 10
        # self.robot.control_dofs_force(propellerTorque, self.rotors_idx)
    
    def compute_rewards(self, rewardsCfg, state, next_state):
        ee_dist = rewardsCfg["ee_dist"] * (next_state[19:22].norm(p=2)-state[19:22].norm(p=2))
        traj_dist = rewardsCfg["traj_dist"] * ((state[:3]-state[22:]).norm(p=2)-rewardsCfg["traj_thresh"])
        is_captured = rewardsCfg["is_captured"] * (1 if state[19:22].norm(p=2) < 0.1 else 0)
        reward = ee_dist + traj_dist + rewardsCfg["time"] + is_captured
        return reward

