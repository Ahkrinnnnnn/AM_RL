import os

from pxr import Usd, UsdPhysics
import AM_RL

def convert_usdc_to_usd(input_path):
    stage = Usd.Stage.Open(input_path)
    root_prim = stage.GetPseudoRoot().GetChildren()[0]
    root_prim_path = root_prim.GetPath()

    if not UsdPhysics.RigidBodyAPI.Get(stage, root_prim_path):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)

    if not UsdPhysics.CollisionAPI.Get(stage, root_prim_path):
        UsdPhysics.CollisionAPI.Apply(root_prim)

    stage.GetRootLayer().Save()

input_usdc = os.path.dirname(os.path.abspath(AM_RL.__file__)) + "/assets/usd/ball/scene.usdc"

convert_usdc_to_usd(input_usdc)
