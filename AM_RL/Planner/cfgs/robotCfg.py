"""Configuration for robots."""

import random
import os

import isaaclab.sim as sim_utils
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

import AM_RL
import AM_RL.Planner.cfgs.CustomFunctions as CustomFunctions

##
# Configuration
##

robotName = "hexacopter370_flying_arm_3"
rotorNames = ["hexacopter370__rotor_" + str(i) + "_joint" for i in range(6)]
rotorLinkNames = ["hexacopter370__rotor_" + str(i) for i in range(6)]
jointNames = ["flying_arm_3__j_base_link_link_1",
    "flying_arm_3__j_link_1_link_2",
    "flying_arm_3__j_link_2_link_3"]
eeName = "flying_arm_3__gripper"
baseLinkName = "hexacopter370__base_link"
objName = "ball"

rootPath = os.path.dirname(os.path.abspath(AM_RL.__file__))

UAM_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/uam",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=rootPath + f"/assets/urdf/{robotName}.urdf",
        usd_dir=rootPath+f"/assets/usd/uam/",
        usd_file_name=f"{robotName}.usd",
        force_usd_conversion=True,
        fix_base=False,
        copy_from_source=True,
        root_link_name=baseLinkName,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=10.0)
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1), rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*": 0.0
        },
    ),
    actuators={
        "propeller_actuator": ImplicitActuatorCfg(
            joint_names_expr=rotorNames,
            stiffness=0.0,
            damping=10.0
        ),
        "joint_actuator": ImplicitActuatorCfg(
            joint_names_expr=jointNames[:2],
            effort_limit=1.0, 
            velocity_limit=10000.0, 
            stiffness=200.0,
            damping=45.0
        ),
        "last_joint_actuator": ImplicitActuatorCfg(
            joint_names_expr=jointNames[2],
            effort_limit=0.3, 
            velocity_limit=10000.0, 
            stiffness=200.0,
            damping=45.0
        )
    }
)

obj_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/objective",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=rootPath+f"/assets/urdf/{objName}.urdf",
        usd_dir=rootPath+f"/assets/usd/ball/",
        usd_file_name=f"{objName}.usd",
        force_usd_conversion=True,
        fix_base=False,
        copy_from_source=True,
        root_link_name="base_link",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0)
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            density=0.01
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        # pos=((random.random()-0.5)*20, (random.random()-0.5)*20, 0.05)
        pos=(0, -3, 0.05)
    )
)

"""Configuration for robots."""
