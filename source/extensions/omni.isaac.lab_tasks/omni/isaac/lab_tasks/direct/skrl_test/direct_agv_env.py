from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import torch
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
)
from omni.isaac.lab.envs import (
    DirectRLEnv,
    DirectRLEnvCfg,
    ViewerCfg,
)
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import (
    Camera,
    CameraCfg,
    ContactSensor,
    ContactSensorCfg,
)
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from PIL import Image
from pxr import UsdGeom

from .agv_cfg import AGV_CFG, AGV_JOINT


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


##
# Scene definition
##

ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class AGVEnvCfg(DirectRLEnvCfg):
    # env
    dt = 1 / 120
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    num_actions = 3
    num_channels = 3
    num_states = 63
    # events = AGVEventCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = AGV_CFG.replace(prim_path=f"{ENV_REGEX_NS}/AGV")

    # camera
    rcam: CameraCfg = CameraCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/rcam_1/Camera",
        data_types=["rgb"],
        spawn=None,
        width=440,
        height=440,
    )

    niro_cfg = RigidObjectCfg(
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

    agv_joint: AGV_JOINT = AGV_JOINT()

    lpin_cfg = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/lpin_1",
        spawn=None,
    )

    rpin_cfg = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/rpin_1",
        spawn=None,
    )

    niro_contact_cfg = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/Niro/de_1",
    )
    agv_contact_cfg = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/mb_1",
    )

    actuated_joint_names = [
        # "jlw",
        # "jrw",
        # "jz",
        "jy",
        "jx",
        # "jr",
        # "jlr",
        # "jrr",
        # "jlpin",
        "jrpin",
    ]

    num_observations = num_channels * rcam.height * rcam.width
    write_image_to_file = False

    # change viewer settings
    viewer = ViewerCfg(eye=(4.0, 0.0, 3.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)


class AGVEnv(DirectRLEnv):
    cfg: AGVEnvCfg

    def __init__(self, cfg: AGVEnvCfg, render_mode: str | None = None, **kwargs):
        # print(1)
        super().__init__(cfg, render_mode, **kwargs)

        # self._MB_LW_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_LW_REV)
        # self._MB_RW_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_RW_REV)
        # self._MB_PZ_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_PZ_PRI)
        # self._PZ_PY_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PZ_PY_PRI)
        # self._PY_PX_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PY_PX_PRI)
        # self._PX_PR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PX_PR_REV)
        # self._PR_LR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PR_LR_REV)
        # self._PR_RR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PR_RR_REV)
        # self._LR_LPIN_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.LR_LPIN_PRI)
        # self._RR_RPIN_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.RR_RPIN_PRI)
        self._RPIN_idx, _ = self._agv.find_bodies("rpin_1")
        self._LPIN_idx, _ = self._agv.find_bodies("lpin_1")
        # self._XY_PRI_idx, _ = self._agv.find_joints([self.cfg.agv_joint.PZ_PY_PRI, self.cfg.agv_joint.PY_PX_PRI])
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self._agv.data.joint_pos
        self.joint_vel = self._agv.data.joint_vel

        self.episode = 1

        self.num_agv_dofs = self._agv.num_joints

        # # buffers for position targets
        self.agv_dof_targets = torch.zeros((self.num_envs, self.num_agv_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_agv_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_agv_dofs), dtype=torch.float, device=self.device)

        # # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self._agv.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # # joint limits
        joint_pos_limits = self._agv.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        self.joint_pos = self._agv.data.joint_pos
        self.joint_vel = self._agv.data.joint_vel

        if len(self.cfg.rcam.data_types) != 1:
            raise ValueError(
                "The camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.rcam.data_types}"
            )

    def close(self):
        # print(2)
        """Cleanup for the environment."""
        super().close()

    def _configure_gym_env_spaces(self):
        # Configure the action and observation spaces for the Gym environment.
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Dict(
            image=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.cfg.rcam.height,
                    self.cfg.rcam.width,
                    self.cfg.num_channels,
                ),
            ),
            value=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions * 13,)),
            critic=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,)),
        )

        if not self.cfg.num_states:
            self.state_space = None
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.num_states,
                    # self.cfg.rcam.height,
                    # self.cfg.rcam.width,
                    # self.cfg.num_channels,
                ),
            )
            # shape=(self.num_states,)
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)
        else:
            self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.num_states,))

        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"],
            self.num_envs,
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space,
            self.num_envs,
        )

    def _setup_scene(self):
        # print(4)
        self._agv = Articulation(self.cfg.robot_cfg)
        self._niro = RigidObject(self.cfg.niro_cfg)
        self._agv_contact = ContactSensor(self.cfg.agv_contact_cfg)
        self._niro_contact = ContactSensor(self.cfg.niro_contact_cfg)
        self._rcam = Camera(self.cfg.rcam)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion and sensors to scene
        self.scene.articulations["agv"] = self._agv
        self.scene.rigid_objects["niro"] = self._niro
        self.scene.sensors["agv_contact"] = self._agv_contact
        self.scene.sensors["niro_contact"] = self._niro_contact
        self.scene.sensors["rcam"] = self._rcam

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.my_visualizer = define_markers()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print(5)
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        if torch.isnan(self.actions).any():
            raise ValueError("Actions contain NaN values.")
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._XY_PRI_idx + self._RR_RPIN_PRI_idx)
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._PY_PX_PRI_idx)
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._PZ_PY_PRI_idx)
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._PY_PX_PRI_idx)
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._RR_RPIN_PRI_idx)
        self._agv.set_joint_effort_target(self.actions, joint_ids=self.actuated_dof_indices)

    def _get_observations(self) -> dict:
        if self.episode == 0:
            self.initial_pin_position()

        data_type = "rgb" if "rgb" in self.cfg.rcam.data_types else "depth"
        tensor = self._rcam.data.output[data_type].clone()[:, :, :, :3]

        # values = torch.cat(
        #     (
        #         self.joint_vel[:, self.actuated_dof_indices],
        #         self.joint_pos[:, self.actuated_dof_indices],
        #     ),
        #     dim=-1,
        # )

        values = self._agv.data.body_state_w[:, self.actuated_dof_indices].view(self.num_envs, self.num_actions * 13)

        if self.cfg.write_image_to_file:
            array = tensor.squeeze(0).cpu().numpy()
            array = (array - array.min()) / (array.max() - array.min()) * 255
            array = array.astype("uint8")  # Convert to uint8 data type
            image = Image.fromarray(array)
            image.save("output_image.png")

        observations = {
            "policy": {
                "value": values,
                "image": (tensor.type(torch.cuda.FloatTensor) / 255.0),
                "critic": self._get_states(),
            },
            "critic": self._get_states(),
        }

        self.episode += 1

        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # self._agv.data.body_pos_w[:, self.actuated_dof_indices].view(self.num_envs, self.num_actions * 3),
                # self._agv.data.body_quat_w[:, self.actuated_dof_indices].view(self.num_envs, self.num_actions * 4),
                # self._agv.data.body_vel_w[:, self.actuated_dof_indices].view(self.num_envs, self.num_actions * 6),
                # self._agv.data.body_ang_vel_w[:, self.actuated_dof_indices].view(self.num_envs, self.num_actions * 6),
                # self._agv.data.body_ang_acc_w[:, self.actuated_dof_indices].view(self.num_envs, self.num_actions * 6),
                self._agv.data.body_state_w[:, self.actuated_dof_indices].view(self.num_envs, self.num_actions * 13),
                # niro
                self._niro.data.body_pos_w[:, 0] - self.scene.env_origins,
                # applied actions (3)
                self.actions,
                # pin
                self.pin_position(True) - self.scene.env_origins,
                self.pin_position(False) - self.scene.env_origins,
                # # hole
                self.hole_position(True) - self.scene.env_origins,
                self.hole_position(False) - self.scene.env_origins,
                # initial values
                self.init_hole_pos - self.scene.env_origins,
                self.init_pin_pos - self.scene.env_origins,
            ),
            dim=-1,
        )
        
        # states = torch.where(torch.isinf(states), torch.tensor(-1.0), states)
        # print(states.shape)

        return states

    def _get_rewards(self) -> torch.Tensor:
        # reward
        rew_pin_r = self.pin_reward(True)
        correct_rew = self.pin_correct(True).int() * 1000

        # penalty
        z_penalty = (
            -self.terminate_z().int()
            * self.euclidean_distance(self.pin_position(True), self.hole_position(True))
            * 1000
        )
        contact_penalty = -self.is_undesired_contacts(self._niro_contact).int() * 0.1

        # sum
        total_reward = rew_pin_r + correct_rew + z_penalty + contact_penalty

        # UP = "\x1b[3A"
        # print(
        #     f"\npin: {round(rew_pin_r[0].item(), 2)} correct: {round(correct_rew[0].item(), 2)} z: {round(z_penalty[0].item(), 2)} contact: {round(contact_penalty[0].item(), 2)} total: {round(total_reward[0].item(), 2)}_\n{UP}\r"
        # )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print(9)
        # self.joint_pos = self._agv.data.root_pos_w
        # self.joint_vel = self._agv.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pin_out_of_hole = self.terminate_z()
        pin_in_hole = self.pin_correct(True)

        return torch.logical_or(pin_in_hole, pin_out_of_hole), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        self.randomize_joints_by_offset(env_ids, (-0.03, 0.03), "agv")
        self.randomize_object_position(env_ids, (-0.05, 0.05), (-0.03, 0.03), "niro")
        self.episode = 0

        # set joint positions with some noise
        # joint_pos, joint_vel = self._agv.data.default_joint_pos.clone(), self._agv.data.default_joint_vel.clone()
        # joint_pos += torch.rand_like(joint_pos) * 0.1
        # self._agv.write_joint_state_to_sim(joint_pos, joint_vel)
        # clear internal buffers
        # self._agv.reset()

    """
    custom functions
    """

    def reset_scene_to_default(self, env_ids: torch.Tensor):
        """Reset the scene to the default state specified in the scene configuration."""
        # rigid bodies
        for rigid_object in self.scene.rigid_objects.values():
            # obtain default and deal with the offset for env origins
            default_root_state = rigid_object.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
            # set into the physics simulation
            rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)
        # articulations
        for articulation_asset in self.scene.articulations.values():
            # obtain default and deal with the offset for env origins
            default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
            # set into the physics simulation
            articulation_asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
            # obtain default joint positions
            default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
            default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
            # set into the physics simulation
            articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    def randomize_joints_by_offset(
        self,
        env_ids: torch.Tensor,
        position_range: tuple[float, float],
        joint_name: str,
        # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        asset: Articulation = self.scene.articulations[joint_name]
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_pos += math_utils.sample_uniform(
            *position_range,
            joint_pos.shape,
            joint_pos.device,
        )
        joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        asset.write_joint_state_to_sim(joint_pos, 0, env_ids=env_ids)

    def randomize_object_position(
        self,
        env_ids: torch.Tensor,
        xy_position_range: tuple[float, float],
        z_position_range: tuple[float, float],
        joint_name: str,
        # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        rigid_object = self.scene.rigid_objects[joint_name]
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += self.scene.env_origins[env_ids]

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

        # set into the physics simulation
        rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)

    def pin_position(self, right: bool = True):
        root_position: RigidObject = self._agv.data.body_pos_w[:, self._RPIN_idx[0] if right else self._LPIN_idx[0], :]
        pin_rel = torch.tensor(
            [0, 0.02 if right else -0.02, 0.479],
            device="cuda:0",
        )  # 0.479
        pin_pos_w = root_position + pin_rel
        return pin_pos_w

    def hole_position(self, right: bool = True):
        # niro: RigidObject = self.scene.rigid_objects["niro"]
        niro_pos = self._niro.data.root_pos_w  # torch.tensor([-0.5000,  0.0000,  1.1000], device="cuda:0")
        hole_rel = torch.tensor(
            [0.455, 0.693 if right else -0.693, 0.0654],
            device="cuda:0",
        )
        hole_pos_w = torch.add(niro_pos, hole_rel)

        # l hole [.455, .693, .0654] 33.5 (1)/ 50.8(2) / 65.4(3) / 75.5(all)
        # niro [-0.5000,  0.0000,  1.1000]
        # lpin [0, .163, .514 / .479]
        return hole_pos_w

    def terminate_z(self) -> torch.Tensor:
        # l_hole_pos_w = hole_positions(env, False)
        r_hole_pos_w = self.hole_position(True)
        # l_pin_pos_w = pin_positions(env, False)
        r_pin_pos_w = self.pin_position(True)

        r_hole_xy = r_hole_pos_w[:, :1]
        r_pin_xy = r_pin_pos_w[:, :1]

        r_hole_z = r_hole_pos_w[:, 2]
        r_pin_z = r_pin_pos_w[:, 2]

        z_condition = r_pin_z >= r_hole_z

        xy_distance = self.euclidean_distance(r_hole_xy, r_pin_xy)
        xy_condition = xy_distance >= 0.01

        return torch.logical_and(z_condition, xy_condition)

    def pin_correct(self, right: bool = True):
        hole_pos_w = self.hole_position(right)
        pin_pos_w = self.pin_position(right)
        distance = self.euclidean_distance(hole_pos_w, pin_pos_w)
        return distance < 0.01

    def euclidean_distance(self, src, dist):
        distance = torch.sqrt(torch.sum((src - dist) ** 2, dim=1))
        return distance

    def pin_reward(self, right: bool = True) -> torch.Tensor:
        hole_pos_w = self.hole_position(right)
        curr_pin_pos_w = self.pin_position(right)
        prev_pin_pos_w = self.prev_pos_w[f"{'r' if right else 'l'}_pin"]

        hole_xy = hole_pos_w[:, 0:1]
        curr_pin_xy = curr_pin_pos_w[:, 0:1]
        curr_xy_distance = self.euclidean_distance(hole_xy, curr_pin_xy)
        curr_xy_rew = (self.init_xy_distance_r - curr_xy_distance)**3

        prev_pin_xy = prev_pin_pos_w[:, 0:1]
        prev_xy_distance = self.euclidean_distance(hole_xy, prev_pin_xy)

        relative_xy_rew = (prev_xy_distance - curr_xy_distance)

        hole_z = hole_pos_w[:, 2]
        curr_pin_z = curr_pin_pos_w[:, 2]
        curr_z_dist = hole_z - curr_pin_z
        curr_z_rew = (self.init_z_distance_r - curr_z_dist)**3

        prev_pin_z = prev_pin_pos_w[:, 2]
        prev_z_dist = hole_z - prev_pin_z

        relative_z_rew = (prev_z_dist - curr_z_dist)*0.3
        self.prev_pos_w[f"{'r' if right else 'l'}_pin"] = curr_pin_pos_w

        reward = curr_xy_rew + curr_z_rew# + relative_xy_rew + relative_z_rew

        UP = "\x1b[3A"
        print( #rxy: {round(relative_xy_rew[0].item(), 3)} rz: {round(relative_z_rew[0].item(), 3)}
            f"\nxy: {curr_xy_rew[0]} z: {curr_z_rew[0]}  rew: {round(reward[0].item(), 3)}_\n{UP}\r"
        )

        return reward

    def pin_direction_reward(self, hole, pin) -> torch.Tensor:
        current_direction = pin - hole
        current_direction = current_direction / torch.norm(current_direction, p=2)
        cosine_similarity = torch.dot(current_direction, self.init_direction)
        return cosine_similarity

    def initial_pin_position(self):
        r_pin_pos_w = self.pin_position(True)
        l_pin_pos_w = self.pin_position(False)
        l_hole_pos_w = self.hole_position(False)
        r_hole_pos_w = self.hole_position(True)

        self.init_distance_l = self.euclidean_distance(l_pin_pos_w, l_hole_pos_w)
        self.init_distance_r = self.euclidean_distance(r_pin_pos_w, r_hole_pos_w)
        self.init_pin_pos = r_pin_pos_w
        self.init_hole_pos = r_hole_pos_w
        self.init_direction = r_hole_pos_w - r_pin_pos_w

        r_hole_z = r_hole_pos_w[:, 2]
        r_pin_z = r_pin_pos_w[:, 2]

        self.init_z_distance_r = r_hole_z - r_pin_z

        r_hole_xy = r_hole_pos_w[:, 0:1]
        r_pin_xy = r_pin_pos_w[:, 0:1]

        self.init_xy_distance_r = self.euclidean_distance(r_hole_xy, r_pin_xy)

        self.prev_pos_w = {
            "r_pin": r_pin_pos_w,
            "l_pin": l_pin_pos_w,
        }

        marker_locations = torch.vstack((
            self.init_hole_pos,
            self.init_pin_pos - torch.tensor([0, 0, 0.479], device="cuda:0"),
        ))
        self.my_visualizer.visualize(marker_locations)

    def is_undesired_contacts(self, sensor: ContactSensor) -> torch.Tensor:
        net_contact_forces: torch.Tensor = sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, 0], dim=-1), dim=1)[0] > 0
        return is_contact

    def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
        world_transform = xformable.ComputeLocalToWorldTransform(0)
        world_pos = world_transform.ExtractTranslation()
        world_quat = world_transform.ExtractRotationQuat()

        px = world_pos[0] - env_pos[0]
        py = world_pos[1] - env_pos[1]
        pz = world_pos[2] - env_pos[2]
        qx = world_quat.imaginary[0]
        qy = world_quat.imaginary[1]
        qz = world_quat.imaginary[2]
        qw = world_quat.real

        return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
