# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObject,
)
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from .dot_dict import DotDict
import omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp as mdp
from omni.isaac.lab.sensors import CameraCfg, Camera, ContactSensorCfg, ContactSensor
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
import torch

##
# Pre-defined configs
##

AGV_JOINT = DotDict(
    dict(
        MB_LW_REV="jlw",
        MB_RW_REV="jrw",
        MB_PZ_PRI="jz",
        PZ_PY_PRI="jy",
        PY_PX_PRI="jx",
        PX_PR_REV="jr",
        PR_LR_REV="jlr",
        PR_RR_REV="jrr",
        LR_LPIN_PRI="jlpin",
        RR_RPIN_PRI="jrpin",
    )
)


##
# Scene definition
##


@configclass
class AGVSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),  # size=(100.0, 100.0)
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/AGV",
        spawn=sim_utils.UsdFileCfg(
            usd_path="./robot/usd/agv/agv_fixed_pin.usd",
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #     rigid_body_enabled=True,
            #     max_linear_velocity=1000.0,
            #     max_angular_velocity=1000.0,
            #     max_depenetration_velocity=100.0,
            #     enable_gyroscopic_forces=True,
            # ),
            # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            #     enabled_self_collisions=True,
            #     solver_position_iteration_count=4,
            #     solver_velocity_iteration_count=0,
            #     sleep_threshold=0.005,
            #     stabilization_threshold=0.001,
            # ),
            activate_contact_sensors=True,
        ),
        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, 0.0),
        #     joint_pos={
        #         AGV_JOINT.MB_LW_REV: 0.0,
        #         AGV_JOINT.MB_RW_REV: 0.0,
        #         AGV_JOINT.MB_PZ_PRI: 0.0,
        #         AGV_JOINT.PZ_PY_PRI: 0.0,
        #         AGV_JOINT.PY_PX_PRI: 0.0,
        #         AGV_JOINT.PX_PR_REV: 0.0,
        #         AGV_JOINT.PR_LR_REV: 0.0,
        #         AGV_JOINT.PR_RR_REV: 0.0,
        #         AGV_JOINT.LR_LPIN_PRI: 0.0,
        #         AGV_JOINT.RR_RPIN_PRI: 0.0,
        #     },
        # ),
        actuators={
            # "wheel_actuator": ImplicitActuatorCfg(
            #     joint_names_expr=[AGV_JOINT.MB_LW_REV, AGV_JOINT.MB_RW_REV],
            #     effort_limit=200.0,
            #     velocity_limit=100.0,
            #     stiffness=0.0,
            #     damping=0.0,
            # ),
            "xyz_actuator": ImplicitActuatorCfg(
                joint_names_expr=[
                    # AGV_JOINT.MB_PZ_PRI,
                    AGV_JOINT.PZ_PY_PRI,
                    AGV_JOINT.PY_PX_PRI,
                ],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=1000.0,
                damping=1000.0,
            ),
            # "px_pr_rev_actuator": ImplicitActuatorCfg(
            #     joint_names_expr=[AGV_JOINT.PX_PR_REV],
            #     effort_limit=100.0,
            #     velocity_limit=100.0,
            #     stiffness=0.0,
            #     damping=0.0,
            # ),
            # "pin_rev_actuator": ImplicitActuatorCfg(
            #     joint_names_expr=[AGV_JOINT.PR_LR_REV, AGV_JOINT.PR_RR_REV],
            #     effort_limit=200.0,
            #     velocity_limit=100.0,
            #     stiffness=0.0,
            #     damping=0.0,
            # ),
            "pin_pri_actuator": ImplicitActuatorCfg(
                joint_names_expr=[
                    # AGV_JOINT.LR_LPIN_PRI,
                    AGV_JOINT.RR_RPIN_PRI
                ],
                effort_limit=200.0,
                velocity_limit=100.0,
                stiffness=100.0,
                damping=100.0,
            ),
        },
    )
    # AGV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    niro = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Niro",
        spawn=sim_utils.UsdFileCfg(
            usd_path="./robot/usd/niro/niro_fixed.usd",
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, 0.0, 1.05),
            # rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    # lcam = CameraCfg(
    #     data_types=["rgb"],
    #     prim_path="{ENV_REGEX_NS}/AGV/lcam_1/Camera",
    #     spawn=None,
    #     height=256,
    #     width=256,
    #     # update_period=0.1,
    # )
    rcam = CameraCfg(
        data_types=["rgb"],
        prim_path="{ENV_REGEX_NS}/AGV/rcam_1/Camera",
        spawn=None,
        height=300,
        width=300,
        # update_period=0.1,
    )

    lpin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AGV/lpin_1",
        spawn=None,
    )

    rpin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AGV/rpin_1",
        spawn=None,
    )

    niro_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Niro/de_1",
    )
    agv_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/AGV/mb_1",
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         # AGV_JOINT.MB_LW_REV,
    #         # AGV_JOINT.MB_RW_REV,
    #         AGV_JOINT.MB_PZ_PRI,
    #         AGV_JOINT.PZ_PY_PRI,
    #         AGV_JOINT.PY_PX_PRI,
    #         AGV_JOINT.PX_PR_REV,
    #         AGV_JOINT.PR_LR_REV,
    #         AGV_JOINT.PR_RR_REV,
    #         AGV_JOINT.LR_LPIN_PRI,
    #         AGV_JOINT.RR_RPIN_PRI,
    #     ],
    #     scale=100.0,
    # )

    joint_pri = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            # AGV_JOINT.MB_PZ_PRI,
            AGV_JOINT.PZ_PY_PRI,
            AGV_JOINT.PY_PX_PRI,
        ],
        scale=100.0,
    )

    # joint_pxpr = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         AGV_JOINT.PX_PR_REV,
    #     ],
    #     scale=100.0,
    # )

    # joint_rev = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         AGV_JOINT.PX_PR_REV,
    #         AGV_JOINT.PR_LR_REV,
    #         AGV_JOINT.PR_RR_REV,
    #     ],
    #     scale=100.0,
    # )
    joint_pin = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            # AGV_JOINT.LR_LPIN_PRI,
            AGV_JOINT.RR_RPIN_PRI,
        ],
        scale=100.0,
    )


def l_cam_rgb(env: ManagerBasedEnv):
    camera: Camera = env.scene["lcam"]
    observations = camera.data.output["rgb"].clone()
    rgb = observations[:, :, :, :3]  # .flatten(start_dim=1)
    return rgb


def r_cam_rgb(env: ManagerBasedEnv):
    camera: Camera = env.scene["rcam"]
    observations = camera.data.output["rgb"].clone()
    rgb = observations[:, :, :, :3]  # .flatten(start_dim=1)
    # grayscale = (0.2989 * rgb[:, :, :, 0] + 
    #              0.5870 * rgb[:, :, :, 1] + 
    #              0.1140 * rgb[:, :, :, 2])
    return rgb


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # lcam = ObsTerm(func=l_cam_rgb)
        rcam = ObsTerm(func=r_cam_rgb)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def initial_pin_position(env, env_ids) -> torch.Tensor:
    r_pin: RigidObject = env.scene["rpin"]
    l_pin: RigidObject = env.scene["lpin"]
    r_pin_rel = torch.tensor([0, .02, .479], device="cuda:0")
    l_pin_rel = torch.tensor([0, -.02, .479], device="cuda:0")
    r_pin.data.update(1)
    r_pin_pos_w = torch.add(r_pin.data.root_pos_w, r_pin_rel)
    l_pin_pos_w = torch.add(l_pin.data.root_pos_w, l_pin_rel)
    l_hole_pos_w = hole_positions(env, False)
    r_hole_pos_w = hole_positions(env, True)
    l_pin_pos_w = pin_positions(env, False)
    r_pin_pos_w = pin_positions(env, True)
    distance_l = torch.sub(l_pin_pos_w, l_hole_pos_w)
    distance_r = torch.sub(r_pin_pos_w, r_hole_pos_w)
    global init_distances_l
    init_distances_l = torch.norm(distance_l, dim=1)
    global init_distances_r
    init_distances_r = torch.norm(distance_r, dim=1)

    # print(f"rinit: {init_distances_r}")


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    # reset_xyz_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 # AGV_JOINT.MB_PZ_PRI,
    #                 AGV_JOINT.PZ_PY_PRI,
    #                 AGV_JOINT.PY_PX_PRI,
    #             ]
    #         ),
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # reset_pins_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 AGV_JOINT.LR_LPIN_PRI,
    #                 AGV_JOINT.RR_RPIN_PRI,
    #             ]
    #         ),
    #         "position_range": (0, 0),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # reset_rev_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[AGV_JOINT.PX_PR_REV]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )

    reset_niro_position = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        # params={
        #     "asset_cfg": SceneEntityCfg("niro"),
        #     "position_range": (-0.25, 0.25),
        #     "velocity_range": (-0.25, 0.25),
        # },
    )

    init_position = EventTerm(
        func=initial_pin_position,
        mode="reset",
    )


def pin_positions(env: ManagerBasedRLEnv, right: bool = True):
    pin: RigidObject = env.scene[f"{'r' if right else 'l'}pin"]
    pin_rel = torch.tensor([0, 0.02 if right else -0.02, .479], device="cuda:0") # 0.479
    pin_pos_w = torch.add(pin.data.root_pos_w, pin_rel)
    return pin_pos_w


def hole_positions(env: ManagerBasedRLEnv, right: bool = True):
    niro: RigidObject = env.scene["niro"]
    niro_pos = (
        niro.data.root_pos_w
    )  # torch.tensor([-0.5000,  0.0000,  1.1000], device="cuda:0")
    hole_rel = torch.tensor([0.455, 0.693 if right else -0.693, 0.0654], device="cuda:0")
    hole_pos_w = torch.add(niro_pos, hole_rel)

    # l hole [.455, .693, .0654] 33.5 (1)/ 50.8(2) / 65.4(3) / 75.5(all)
    # niro [-0.5000,  0.0000,  1.1000]
    # lpin [0, .163, .514 / .479]
    return hole_pos_w


def r_pin_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    r_hole_pos_w = hole_positions(env, True)
    r_pin_pos_w = pin_positions(env, True)
    distance_r = torch.sub(r_pin_pos_w, r_hole_pos_w)
    distances = torch.norm(distance_r, dim=1)
    # print(distances)

    # global init_distances_l
    # rew = torch.zeros_like(distances)

    # for i in range(distances.size(0)):
    #     if distances[i] > init_distances_l[i]:
    #         rew[i] = init_distances_l[i] - distances[i]
    #         # rew[i] = -1 / (distances[i] + 1e-5)
    #     else:
    #         rew[i] = torch.exp(-distances[i])
    # rew = torch.exp(-distances)
    rew = torch.sub(init_distances_r, distances)

    if rew < 0:
        rew *= 100

    return rew


def l_pin_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    l_hole_pos_w = hole_positions(env, False)
    l_pin_pos_w = pin_positions(env, False)
    distance_l = torch.sub(l_pin_pos_w, l_hole_pos_w)
    distances = torch.norm(distance_l, dim=1)
    rew = torch.exp(-distances)
    return rew

def penalty_z(env) -> torch.Tensor:
    # l_hole_pos_w = hole_positions(env, False)
    r_hole_pos_w = hole_positions(env, True)
    # l_pin_pos_w = pin_positions(env, False)
    r_pin_pos_w = pin_positions(env, True)

    r_hole_x = r_hole_pos_w[:, 0]
    r_hole_y = r_hole_pos_w[:, 1]
    r_hole_z = r_hole_pos_w[:, 2]
    
    r_pin_x = r_pin_pos_w[:, 0]
    r_pin_y = r_pin_pos_w[:, 1]
    r_pin_z = r_pin_pos_w[:, 2]

    z_condition = r_pin_z >= r_hole_z

    xy_distance = torch.sqrt((r_pin_x - r_hole_x) ** 2 + (r_pin_y - r_hole_y) ** 2)
    xy_condition = xy_distance >= 0.01

    if z_condition and xy_condition:
        return True
    else:
        return False

    # combined_condition = z_condition & xy_condition
    # penalty = torch.where(combined_condition, torch.tensor(xy_distance * 10, device='cuda:0'), torch.tensor(0.0, device='cuda:0'))

    # return penalty


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    r_pin = RewTerm(func=r_pin_reward, weight=1.0)
    # l_pin = RewTerm(func=l_pin_reward, weight=3.0)
    # r_pin_z = RewTerm(func=penalty_z, weight=-3)

    
    # agv_undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("agv_contact"), 
    #         "threshold": 1.0,
    #     },
    # )
    
    niro_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "niro_contact",
                # body_names=".*THIGH"
            ), 
            "threshold": 1.0,
        },
    )

    # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["jr"]), "target": 0.0},
    # )
    # (4) Shaping tasks: lower cart velocity
    # pri_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[
    #                 # AGV_JOINT.MB_PZ_PRI,
    #                 AGV_JOINT.PZ_PY_PRI,
    #                 AGV_JOINT.PY_PX_PRI,
    #                 # AGV_JOINT.LR_LPIN_PRI,
    #                 AGV_JOINT.RR_RPIN_PRI,
    #             ]
    #         )
    #     },
    # )
    # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )


def termination_accel(
    env: ManagerBasedRLEnv,
    limit_acc: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    acc = asset.data.body_lin_acc_w
    magnitudes = torch.norm(acc, dim=2)
    mean_magnitude = magnitudes.mean()
    return mean_magnitude > limit_acc


def is_pin_in_hole(env: ManagerBasedRLEnv) -> torch.Tensor:
    r_hole_pos_w = hole_positions(env, True)
    r_pin_pos_w = pin_positions(env, True)
    distance_r = torch.sub(r_pin_pos_w, r_hole_pos_w)
    distances = torch.norm(distance_r, dim=1)

def out_of_limit(env: ManagerBasedRLEnv) -> torch.Tensor:
    # l_hole_pos_w = hole_positions(env, False)
    r_hole_pos_w = hole_positions(env, True)
    # l_pin_pos_w = pin_positions(env, False)
    r_pin_pos_w = pin_positions(env, True)

    distance_r = torch.sub(r_pin_pos_w[:, :2], r_hole_pos_w[:, :2])
    distances = torch.norm(distance_r, dim=1)
    return distances > 0.05


def undesired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0
    return is_contact


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # niro_bad_orientation = DoneTerm(
    #     func=termination_accel,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "niro",
    #             # joint_names=[AGV_JOINT.MB_PZ_PRI],
    #         ),
    #         "limit_acc": 1.0,
    #     },
    # )
    
    # pole_out_of_bounds = DoneTerm(func=out_of_limit)
    # pole_contacts = DoneTerm(func=undesired_contacts)

    pin_in_hole = DoneTerm(func=penalty_z)


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


##
# Environment configuration
##


@configclass
class AGVEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: AGVSceneCfg = AGVSceneCfg(num_envs=4, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
