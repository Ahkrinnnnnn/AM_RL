"""Configuration for a UAM robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

robotName = "hexacopter370_flying_arm_3"
rotorNames = ["hexacopter370__rotor_" + str(i) + "_joint" for i in range(6)]
jointNames = ["flying_arm_3__j_base_link_link_1",
    "flying_arm_3__j_link_1_link_2",
    "flying_arm_3__j_link_2_link_3"]
eeName = "flying_arm_3__gripper"

UAM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"assets/urdf/{robotName}.urdf",
        usd_dir=f"assets/usd/",
        usd_file_name=f"{robotName}.usd"
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1), rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            jointNames[0]: (0, 0, 0),
            jointNames[1]: (0.132, 0, 0), 
            jointNames[2]: (0.207, 0, 0)
        },
    ),
    actuators={
        "propeller_actuator": ImplicitActuatorCfg(
            joint_names_expr=rotorNames,
        ),
        "joint_actuator": ImplicitActuatorCfg(
            joint_names_expr=jointNames[:2],
            effort_limit=1, 
            velocity_limit=10000.0, 
        ),
        "last_joint_actuator": ImplicitActuatorCfg(
            joint_names_expr=jointNames[-1],
            effort_limit=0.3, 
            velocity_limit=10000.0, 
        )
    }
)

"""Configuration for a simple UAM robot."""
