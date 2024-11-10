import numpy as np
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp as mdp
import torch
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import ManagerTermBase as TermBase
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.direct.skrl_test.agv_cfg import AGV_CFG, AGV_JOINT
import omni.isaac.core.utils.stage as stage_utils
from pxr import Gf, UsdGeom
import random

ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class AGVSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),  # size=(100.0, 100.0)
    )

    agv: ArticulationCfg = AGV_CFG.replace(prim_path=f"{ENV_REGEX_NS}/AGV")

    niro = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/Niro",
        spawn=sim_utils.UsdFileCfg(
            usd_path="./robot/usd/niro/niro_fixed.usd",
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, 0.0, 1.05),
            # rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    rcam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/AGV/rcam_1/Camera",
        data_types=["rgb"],
        spawn=None,
        height=300,
        width=300,
        # update_period=0.1,
    )

    niro_contact = ContactSensorCfg(prim_path=f"{ENV_REGEX_NS}/Niro/de_1")
    agv_contact = ContactSensorCfg(prim_path=f"{ENV_REGEX_NS}/AGV/mb_1")

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
        asset_name="agv",
        joint_names=[
            # AGV_JOINT.MB_PZ_PRI,
            AGV_JOINT.PZ_PY_PRI,
            AGV_JOINT.PY_PX_PRI,
        ],
        scale=20.0,
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
        asset_name="agv",
        joint_names=[
            # AGV_JOINT.LR_LPIN_PRI,
            AGV_JOINT.RR_RPIN_PRI,
        ],
        scale=30.0,
    )


@configclass
class TheiaTinyObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny Transformer"""

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("rcam"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cddsv",
                "model_device": "cuda:0",
            },
        )
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("agv")})
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("agv")})
        # actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.enable_corruption = True
        #     self.concatenate_terms = True

    policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()


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


def randomize_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, 0, env_ids=env_ids)


def randomize_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    xy_position_range: tuple[float, float],
    z_position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
):
    rigid_object = env.scene.rigid_objects[asset_cfg.name]
    # obtain default and deal with the offset for env origins
    default_root_state = rigid_object.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    xy_low, xy_high = xy_position_range
    z_low, z_high = z_position_range

    # Random offsets for X and Y coordinates
    xy_random_offsets = torch.tensor(
        np.random.uniform(xy_low, xy_high, size=(default_root_state.shape[0], 2)),  # For X and Y only
        dtype=default_root_state.dtype,
        device=default_root_state.device,
    )

    # Random offsets for Z coordinate
    z_random_offsets = torch.tensor(
        np.random.uniform(z_low, z_high, size=(default_root_state.shape[0], 1)),  # For Z only
        dtype=default_root_state.dtype,
        device=default_root_state.device,
    )

    # Apply random offsets to the X, Y, and Z coordinates
    default_root_state[:, 0:2] += xy_random_offsets  # Apply to X and Y coordinates
    default_root_state[:, 2:3] += z_random_offsets  # Apply to Z coordinate

    # set into the physics simulatio
    rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)


class PinRewBase:
    def pin_positions(self, right: bool = True, env_ids=None):
        pin_idx = self._env.scene.articulations["agv"].find_bodies("rpin_1" if right else "lpin_1")[0]
        pin_root_pos = self._env.scene.articulations["agv"].data.body_pos_w[env_ids, pin_idx, :]
        pin_rel = torch.tensor([0, 0.02 if right else -0.02, 0.45], device="cuda:0")  # 0.479
        pin_pos_w = torch.add(pin_root_pos, pin_rel)
        return pin_pos_w.squeeze(1)

    def hole_positions(self, right: bool = True, env_ids=None):
        niro: RigidObject = self._env.scene.rigid_objects["niro"]
        niro_pos = niro.data.root_pos_w[env_ids]
        hole_rel = torch.tensor([0.455, 0.693 if right else -0.693, 0.0654], device="cuda:0")
        hole_pos_w = torch.add(niro_pos, hole_rel)
        return hole_pos_w.squeeze(1)

    def pin_velocities(self, right: bool = True, env_ids=None):
        pin_idx = self._env.scene.articulations["agv"].find_bodies("rpin_1" if right else "lpin_1")[0]
        pin_vel_w = self._env.scene.articulations["agv"].data.body_vel_w[env_ids, pin_idx, :]
        pin_lv = pin_vel_w.squeeze(1)[..., :3]
        pin_v_norm = torch.norm(pin_lv, dim=-1)
        return pin_v_norm


class pin_pos_reward(TermBase, PinRewBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewTerm):
        super().__init__(cfg, env)
        self.init_pin_pos = all_pin_positions(env, cfg.params["right"])
        self.init_hole_pos = all_hole_positions(env, cfg.params["right"])
        self.init_distance = euclidean_distance(self.init_pin_pos, self.init_hole_pos)

    def reset(self, env_ids: torch.Tensor):
        pin_pos_w = self.pin_positions(self.cfg.params["right"], env_ids)
        hole_pos_w = self.hole_positions(self.cfg.params["right"], env_ids)

        self.init_distance[env_ids] = euclidean_distance(pin_pos_w, hole_pos_w)
        self.init_pin_pos[env_ids] = pin_pos_w
        self.init_hole_pos[env_ids] = hole_pos_w

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_pos_w = all_pin_positions(env, right)
        distance = euclidean_distance(pin_pos_w, self.init_hole_pos)

        curr_pin_pos_w = pin_pos_w
        curr_pin_to_hole = distance

        init_pin_to_cur_pin = euclidean_distance(self.init_pin_pos, curr_pin_pos_w)

        dist3 = init_pin_to_cur_pin + curr_pin_to_hole - self.init_distance
        rew = dist3**3

        xyz_rew = (self.init_distance - curr_pin_to_hole) ** 3

        reward = xyz_rew - rew
        return reward


class pin_vel_reward(TermBase, PinRewBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_vel_w = all_pin_velocities(env, right)
        return pin_vel_w**2


class pin_torque_reward(TermBase, PinRewBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        joint_idx = self._env.scene.articulations["agv"].find_joints(
            [
                AGV_JOINT.RR_RPIN_PRI if right else AGV_JOINT.LR_LPIN_PRI,
                AGV_JOINT.PZ_PY_PRI,
                AGV_JOINT.PY_PX_PRI,
            ]
        )[0]
        torques = self._env.scene.articulations["agv"].data.applied_torque[:, joint_idx]
        return torch.sum(torques**2, dim=1)


class draw_lines(TermBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_list = all_pin_positions(env, right).tolist()
        hole_list = all_hole_positions(env, right).tolist()
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        draw.draw_lines(pin_list, hole_list, [(1, 1, 1, 1)] * self.num_envs, [5] * self.num_envs)
        return torch.zeros(self.num_envs, device="cuda:0").bool()


class pin_correct_reward(TermBase, PinRewBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewTerm):
        super().__init__(cfg, env)
        self.init_hole_pos = all_hole_positions(env, cfg.params["right"])
        self.init_pin_pos = all_pin_positions(env, cfg.params["right"])

    def reset(self, env_ids: torch.Tensor):
        self.init_hole_pos[env_ids] = self.hole_positions(self.cfg.params["right"], env_ids)
        self.init_pin_pos[env_ids] = self.pin_positions(self.cfg.params["right"], env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        # pin_pos = all_pin_positions(env, right)
        # pin_vel = all_pin_velocities(env, right)

        # pin_xy_pos = pin_pos[..., :2]
        # pin_z_pos = pin_pos[..., 2]
        # hole_xy_pos = self.init_hole_pos[..., :2]
        # hole_z_pos = self.init_hole_pos[..., 2]

        # xy_distance = euclidean_distance(hole_xy_pos, pin_xy_pos)
        # xy_correct = xy_distance < 0.01
        # z_correct = torch.logical_and(hole_z_pos + 0.02 > pin_z_pos, pin_z_pos > hole_z_pos - 0.01)

        # distance = euclidean_distance(self.init_hole_pos, pin_pos)
        # pos_correct = torch.logical_and(xy_correct, z_correct)

        # reward = pos_correct.int() * torch.clamp(1 / pin_vel, max=10) * torch.clamp(1 / distance, max=10)
        reward = pin_correct(env, right).int() * euclidean_distance(self.init_hole_pos, self.init_pin_pos)
        return reward


class pin_wrong_reward(TermBase, PinRewBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        penalty = pin_wrong(env, right)
        reward = penalty.int()
        return reward


def randomize_color(env: ManagerBasedEnv, env_ids: torch.Tensor):
    object_names = ["AGV", "Niro"]
    material_names = ["OmniSurfaceLite", "material_silver"]
    property_names = [
        "Shader.inputs:diffuse_reflection_color",
        "Shader.inputs:diffuse_color_constant",
    ]
    stage = stage_utils.get_current_stage()

    for idx, object_name in enumerate(object_names):
        for env_id in env_ids:
            color = Gf.Vec3f(random.random(), random.random(), random.random())
            color_spec = stage.GetAttributeAtPath(
                f"/World/envs/env_{env_id}/{object_name}/Looks/{material_names[idx]}/{property_names[idx]}"
            )
            color_spec.Set(color)


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_xyz_position = EventTerm(
        func=randomize_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "agv",
                joint_names=[
                    # AGV_JOINT.MB_PZ_PRI,
                    AGV_JOINT.PZ_PY_PRI,
                    AGV_JOINT.PY_PX_PRI,
                ],
            ),
            "position_range": (-0.05, 0.05),
        },
    )

    reset_pin_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "agv",
                joint_names=[
                    # AGV_JOINT.LR_LPIN_PRI,
                    AGV_JOINT.RR_RPIN_PRI,
                ],
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )

    # reset_rev_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[AGV_JOINT.PX_PR_REV]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )

    randomize_color = EventTerm(
        func=randomize_color,
        mode="reset",
    )

    reset_niro_position = EventTerm(
        func=randomize_object_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("niro"),
            "xy_position_range": (-0.05, 0.05),
            "z_position_range": (-0.03, 0.03),
        },
    )


def all_pin_positions(env: ManagerBasedRLEnv, right: bool = True):
    pin_idx = env.scene.articulations["agv"].find_bodies("rpin_1" if right else "lpin_1")[0]
    pin_root_pos = env.scene.articulations["agv"].data.body_pos_w[:, pin_idx, :]
    pin_rel = torch.tensor([0, 0.02 if right else -0.02, 0.45], device="cuda:0")  # 0.479
    pin_pos_w = torch.add(pin_root_pos, pin_rel)
    return pin_pos_w.squeeze(1)


def all_hole_positions(env: ManagerBasedRLEnv, right: bool = True):
    niro: RigidObject = env.scene.rigid_objects["niro"]
    niro_pos = niro.data.root_pos_w
    hole_rel = torch.tensor([0.455, 0.693 if right else -0.693, 0.0654], device="cuda:0")
    hole_pos_w = torch.add(niro_pos, hole_rel)
    return hole_pos_w.squeeze(1)


def all_pin_velocities(env: ManagerBasedRLEnv, right: bool = True, env_id=None):
    pin_idx = env.scene.articulations["agv"].find_joints(AGV_JOINT.RR_RPIN_PRI if right else AGV_JOINT.LR_LPIN_PRI)[0]
    pin_vel_w = env.scene.articulations["agv"].data.body_vel_w[:, pin_idx, :]

    pin_lv = pin_vel_w.squeeze(1)[..., :3]
    pin_v_norm = torch.norm(pin_lv, dim=-1)
    return pin_v_norm


def euclidean_distance(src, dist):
    distance = torch.sqrt(torch.sum((src - dist) ** 2, dim=src.ndim - 1) + 1e-8)
    return distance


def power_reward(reward) -> torch.Tensor:
    return torch.where(reward < 0, -((reward - 1) ** 2), (reward + 1) ** 2)


@configclass
class RewardsCfg:
    # alive = RewTerm(func=mdp.is_alive, weight=1)
    # terminating = RewTerm(func=mdp.is_terminated, weight=-1000)

    r_pin_pos = RewTerm(func=pin_pos_reward, weight=1000, params={"right": True})
    # r_pin_vel = RewTerm(func=pin_vel_reward, weight=-500, params={"right": True})
    r_pin_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2e-7, params={"asset_cfg": SceneEntityCfg("agv")})
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3, params={"asset_cfg": SceneEntityCfg("agv")})
    # r_pin_torque = RewTerm(func=pin_torque_reward, weight=-1e-6, params={"right": True})
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3, params={"asset_cfg": SceneEntityCfg("agv")})
    # r_pin_correct = RewTerm(func=pin_correct_reward, weight=1e6, params={"right": True})
    # l_pin = RewTerm(func=l_pin_reward, weight=3.0)
    # r_pin_xy = RewTerm(func=r_pin_xy, weight=1.0)
    # r_pin_z = RewTerm(func=r_pin_z, weight=1.0)

    # agv_undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("agv_contact"),
    #         "threshold": 1.0,
    #     },
    # )

    r_pin_wrong = RewTerm(func=pin_wrong_reward, weight=-100, params={"right": True})

    niro_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "niro_contact",
                # body_names=".*THIGH"
            ),
            "threshold": 1.0,
        },
    )

    # niro_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-5e-5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("niro_contact"),
    #         "threshold": 1.0,
    #     },
    # )

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


def pin_correct(env, right: bool = True) -> torch.Tensor:
    hole_pos_w = all_hole_positions(env, right)
    pin_pos_w = all_pin_positions(env, right)
    distance = euclidean_distance(hole_pos_w, pin_pos_w)

    pin_pos = distance < 0.005
    pin_vel = all_pin_velocities(env, right) < 0.001
    pin_correct = torch.logical_and(pin_pos, pin_vel)

    return pin_correct.squeeze(0)


def pin_wrong(env, right: bool = True) -> torch.Tensor:
    hole_pos_w = all_hole_positions(env, right)
    pin_pos_w = all_pin_positions(env, right)

    hole_y = hole_pos_w[:, 1]
    pin_y = pin_pos_w[:, 1]
    y_distance = torch.abs(hole_y - pin_y)

    hole_z = hole_pos_w[:, 2]
    pin_z = pin_pos_w[:, 2]

    z_condition = pin_z >= hole_z - 0.02
    y_condition = y_distance >= 0.01

    return torch.logical_and(z_condition, y_condition)


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

    # pole_out_of_bounds = DoneTerm(func=pin_wrong, params={"right": True})
    # pole_contacts = DoneTerm(func=undesired_contacts)

    # pin_correct = DoneTerm(func=pin_correct, params={"right": True})
    draw_lines = DoneTerm(func=draw_lines, params={"right": True})


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


@configclass
class AGVEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: AGVSceneCfg = AGVSceneCfg(num_envs=4, env_spacing=3.0)
    # Basic settings
    observations: TheiaTinyObservationCfg = TheiaTinyObservationCfg()
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
        self.decimation = 4
        self.episode_length_s = 6
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # self.viewer.lookat = (0.0, 0.0, 2.5)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
