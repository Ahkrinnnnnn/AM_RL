import gymnasium as gym

gym.register(
    id="Isaac-UAM-Catch-Plan-v0",
    entry_point="AM_RL.Planner.cfgs.envCfg:CustomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "AM_RL.Planner.cfgs.envCfg:UamEnvCfg",
        "sb3_cfg_entry_point": "AM_RL.Planner.cfgs:TD3_cfg.yaml"
    }
)
