import gymnasium as gym
from . import agents
from .direct_agv_env import AGVEnvCfg


gym.register(
    id="Isaac-AGV-Direct",
    entry_point="omni.isaac.lab_tasks.direct.skrl_test.direct_agv_env:AGVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AGVEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
