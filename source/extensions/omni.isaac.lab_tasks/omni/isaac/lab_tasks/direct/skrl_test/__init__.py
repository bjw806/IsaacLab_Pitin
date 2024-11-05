import gymnasium as gym
from . import agents
from .direct_agv_env import AGVEnvCfg


gym.register(
    id="Isaac-AGV-Direct",
    entry_point=f"{__name__}.direct_agv_env:AGVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_agv_env:AGVEnv",
    },
)

gym.register(
    id="Isaac-AGV-Managed",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manage_agv_env:AGVEnvCfg",
    },
)
