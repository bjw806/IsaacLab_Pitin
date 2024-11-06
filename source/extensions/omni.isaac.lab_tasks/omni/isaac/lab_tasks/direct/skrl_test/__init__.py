import gymnasium as gym
from . import agents


gym.register(
    id="Isaac-AGV-Direct",
    entry_point=f"{__name__}.direct_agv_env:AGVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_agv_env:AGVEnvCfg",
    },
)

gym.register(
    id="Isaac-AGV-Managed",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manage_agv_env:AGVEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)
