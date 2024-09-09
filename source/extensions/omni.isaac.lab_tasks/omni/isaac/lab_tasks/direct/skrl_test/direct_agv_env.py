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
from omni.isaac.lab.sensors import (
    Camera,
    CameraCfg,
    ContactSensor,
    ContactSensorCfg,
    save_images_to_file,
)
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass

from .agv_cfg import AGV_CFG, AGV_JOINT
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from PIL import Image

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
    num_actions = 1
    num_channels = 3
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = AGV_CFG.replace(prim_path=f"{ENV_REGEX_NS}/AGV")

    # camera
    rcam: CameraCfg = CameraCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/rcam_1/Camera",
        data_types=["rgb"],
        spawn=None,
        width=128,
        height=128,
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
    

    num_observations = num_channels * rcam.height * rcam.width
    write_image_to_file = False

    # change viewer settings
    viewer = ViewerCfg(eye=(4.0, 0.0, 3.0))

    # scene
    # scene: AGVSceneCfg = AGVSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)


class AGVEnv(DirectRLEnv):
    cfg: AGVEnvCfg

    def __init__(self, cfg: AGVEnvCfg, render_mode: str | None = None, **kwargs):
        # print(1)
        super().__init__(cfg, render_mode, **kwargs)

        self._MB_LW_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_LW_REV)
        self._MB_RW_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_RW_REV)
        self._MB_PZ_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_PZ_PRI)
        self._PZ_PY_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PZ_PY_PRI)
        self._PY_PX_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PY_PX_PRI)
        self._PX_PR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PX_PR_REV)
        self._PR_LR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PR_LR_REV)
        self._PR_RR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PR_RR_REV)
        self._LR_LPIN_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.LR_LPIN_PRI)
        self._RR_RPIN_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.RR_RPIN_PRI)

        self.action_scale = self.cfg.action_scale

        # self.joint_pos = self._agv.data.joint_pos
        # self.joint_vel = self._agv.data.joint_vel

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
        # print(3)
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.cfg.rcam.height,
                self.cfg.rcam.width,
                self.cfg.num_channels,
            ),
        )
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.cfg.rcam.height,
                    self.cfg.rcam.width,
                    self.cfg.num_channels,
                ),
            )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

    def _setup_scene(self):
        # print(4)
        self._agv = Articulation(self.cfg.robot_cfg)
        self._niro = RigidObject(self.cfg.niro_cfg)
        self._rpin = RigidObject(self.cfg.rpin_cfg)
        self._lpin = RigidObject(self.cfg.lpin_cfg)
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
        self.scene.rigid_objects["rpin"] = self._rpin
        self.scene.rigid_objects["lpin"] = self._lpin
        self.scene.sensors["agv_contact"] = self._agv_contact
        self.scene.sensors["niro_contact"] = self._niro_contact
        self.scene.sensors["rcam"] = self._rcam

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print(5)
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # print(6)
        self._agv.set_joint_effort_target(self.actions, joint_ids=self._PZ_PY_PRI_idx)
        self._agv.set_joint_effort_target(self.actions, joint_ids=self._PY_PX_PRI_idx)
        self._agv.set_joint_effort_target(self.actions, joint_ids=self._RR_RPIN_PRI_idx)

    def _get_observations(self) -> dict:
        # print(7)
        data_type = "rgb" if "rgb" in self.cfg.rcam.data_types else "depth"
        tensor = self._rcam.data.output[data_type].clone()[:,:,:,:3]
        #tensor = torch.nn.functional.interpolate(tensor, size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        observations = {"policy": tensor.type(torch.cuda.FloatTensor).view(1, -1)}

        if self.cfg.write_image_to_file:
            # save_images_to_file(observations["policy"], f"agv_{data_type}.png")
            tensor = observations["policy"].squeeze(0)  # Shape now [512, 512, 4]
            array = tensor.cpu().numpy()
            array = (array - array.min()) / (array.max() - array.min()) * 255
            array = array.astype('uint8')  # Convert to uint8 data type
            image = Image.fromarray(array)
            image.save('output_image.png')

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # print(8)
        rew_alive = 1.0 - self.reset_terminated.float()
        rew_termination = self.reset_terminated.float()
        rew_pin_r = self.pin_reward(True)

        total_reward = rew_alive*0.01 + rew_termination + rew_pin_r
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print(9)
        # self.joint_pos = self._agv.data.root_pos_w
        # self.joint_vel = self._agv.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pin_in_hole = self.terminate_z()

        return pin_in_hole, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # print(10)
        if env_ids is None:
            env_ids = self._agv._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self._agv.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._agv.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._agv.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.randomize_joints_by_offset(env_ids, (-0.05, 0.05), "agv")
        self.randomize_object_position(env_ids, (-0.05, 0.05), (-0.03, 0.03), "niro")

        self.initial_pin_position()

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
        pin: RigidObject = self.scene.rigid_objects[f"{'r' if right else 'l'}pin"]
        pin_rel = torch.tensor(
            [0, 0.02 if right else -0.02, 0.479],
            device="cuda:0",
        )  # 0.479
        pin_pos_w = torch.add(pin.data.root_pos_w, pin_rel)
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

        r_hole_xy = r_hole_pos_w[:, 0:1]
        r_pin_xy = r_pin_pos_w[:, 0:1]

        r_hole_z = r_hole_pos_w[:, 2]
        r_pin_z = r_pin_pos_w[:, 2]

        z_condition = r_pin_z >= r_hole_z

        xy_distance = torch.norm(torch.sub(r_hole_xy, r_pin_xy), dim=1)
        xy_condition = xy_distance >= 0.01

        result = torch.where(z_condition & xy_condition, torch.tensor(True), torch.tensor(False))
        return result

    def pin_reward(self, right: bool = True) -> torch.Tensor:
        hole_pos_w = self.hole_position(right)
        pin_pos_w = self.pin_position(right)
        diff = torch.sub(pin_pos_w, hole_pos_w)
        dist = torch.norm(diff, dim=1)
        rew = torch.sub(self.init_distance_r if right else self.init_distance_l, dist)
        return rew

    def initial_pin_position(self) -> torch.Tensor:
        r_pin: RigidObject = self.scene["rpin"]
        l_pin: RigidObject = self.scene["lpin"]
        r_pin_rel = torch.tensor([0, 0.02, 0.479], device="cuda:0")
        l_pin_rel = torch.tensor([0, -0.02, 0.479], device="cuda:0")
        r_pin.data.update(1 / 120)
        r_pin_pos_w = torch.add(r_pin.data.root_pos_w, r_pin_rel)
        l_pin_pos_w = torch.add(l_pin.data.root_pos_w, l_pin_rel)
        l_hole_pos_w = self.hole_position(False)
        r_hole_pos_w = self.hole_position(True)
        l_pin_pos_w = self.pin_position(False)
        r_pin_pos_w = self.pin_position(True)
        distance_l = torch.sub(l_pin_pos_w, l_hole_pos_w)
        distance_r = torch.sub(r_pin_pos_w, r_hole_pos_w)

        self.init_distance_l = torch.norm(distance_l, dim=1)
        self.init_distance_r = torch.norm(distance_r, dim=1)

        r_hole_z = r_hole_pos_w[:, 2]
        r_pin_z = r_pin_pos_w[:, 2]

        self.init_z_distance_r = torch.sub(r_hole_z, r_pin_z)

        r_hole_xy = r_hole_pos_w[:, 0:1]
        r_pin_xy = r_pin_pos_w[:, 0:1]

        self.init_xy_distance_r = torch.norm(torch.sub(r_hole_xy, r_pin_xy), dim=1)

        # print(f"rinit: {init_distances_r}")

    def cam_rgb(self, right: bool = True):
        camera: Camera = self.scene[f"{'r' if right else 'l'}cam"]
        observations = camera.data.output["rgb"].clone()
        rgb = observations[:, :, :, :3]  # .flatten(start_dim=1)
        # grayscale = (0.2989 * rgb[:, :, :, 0] +
        #              0.5870 * rgb[:, :, :, 1] +
        #              0.1140 * rgb[:, :, :, 2])
        return rgb
