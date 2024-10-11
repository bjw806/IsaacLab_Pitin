# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv,DirectRLEnv

# from omni.isaac.lab_tasks.manager_based.classic.cartpole.agv_env_cfg import AGVEnvCfg, pin_positions, hole_positions
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.utils.math import quat_from_angle_axis
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab_tasks.direct.skrl_test.direct_agv_env import AGVEnvCfg, AGVEnv


def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

def main():
    """Main function."""
    # create environment configuration
    env_cfg = AGVEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = AGVEnv(cfg=env_cfg)
    my_visualizer = define_markers()


    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            marker_locations = torch.vstack((env.hole_position(True), env.pin_position(True)))
            my_visualizer.visualize(marker_locations)


            # sample random actions
            joint_efforts = torch.randn_like(env.actions)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            # import time
            # time.sleep(0.5)
            # print(rew)
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()