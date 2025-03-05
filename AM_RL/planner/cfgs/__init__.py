import gymnasium as gym
from AM_RL.planner.cfgs import envCfg

gym.register(
    id="Isaac-UAM-Catch-Plan-v0",
    entry_point="AM_RL.planner.cfgs.envCfg:CustomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": envCfg.UamEnvCfg,
        "sb3_cfg_entry_point": "AM_RL.planner.cfgs:TD3_cfg.yaml"
    }
)
